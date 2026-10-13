import numpy as np
import pandas as pd
import logging
from game_time import GameTime
from time import time
from copy import copy
from scipy.spatial import Delaunay
import json
import pitch_control as pc
       

###################################################################################################
############################################ EVENT DETECTION ######################################
###################################################################################################

def event_detection(data_input):
    
    """
    This routine is the main function to generate events after possession_model is run. It sequentially executes different functions
    that correspond to different aspects of auto-eventing, namely gains/losses, dead ball events, shots/saves, score, temporal event logic,
    duels, team possession, possession logic, postprocess, pressures and defensive modes.

    :param data_input: Dictionary containing the output of possession_model
    :return: Dictionary containing all the data generated, including events, db_intervals, log_incidence, team possession, 
    defensive/offensive modes (if applicable), pressures (if applicable)
    """ 

    raw_possession_model = data_input['models']['raw_possession_model'].copy()
    db_intervals = data_input['models']['db_intervals'].copy()
    params = data_input['params']
    hyperparams = params['hyperparams']
    ball_data = params['ball_data'].copy().set_index('frame')
    inplay_triggers = {'kickoff':data_input['models']['kickoff_ingame'],
                      'penalty':data_input['models']['penalty_ingame']}
    
    
    ball_dead = raw_possession_model.query("event_name=='ball dead'")
    model = raw_possession_model.drop(ball_dead.index).reset_index().rename(columns={'index':'index_with_nan'}).copy()
    log_incidence = data_input['log_incidence'].copy()
    
    if 'ball_changed_speed' not in model.columns:
        model['ball_changed_speed'] = model['change_in_speed'].apply(lambda x: abs(x) > hyperparams['ball_change_speed'] if not np.isnan(x) else False)
    if 'ball_changed_direction' not in model.columns:
        model['ball_changed_direction'] = model['change_in_direction'].apply(lambda x: x < hyperparams['ball_change_direction'] if not np.isnan(x) else False)

    model['loss_gain'] = model['event_name'] 
    model['is_goalkeeper'] = False
    model['is_goalkeeper'] = model['player_id'].apply(lambda x: not set(x.split(';')).isdisjoint(params['goalkeeper_ids']))
    model['event_outcome'] = ''
    
    t0 = time()

    ## Establish whether during a gain-possession-loss sequence the involved players touch the ball or not
    model, goalie_frames, log_incidence = set_gains_losses(model, params, log_incidence)
    
    ## Establish dead ball events
    model, log_incidence, db_intervals = set_dead_ball_events(model, db_intervals, ball_data, inplay_triggers, params, log_incidence)
    
    ## Set saves and shots
    model = set_saves_shots(model, ball_dead, goalie_frames, ball_data, params)

    ## Set changes in score, as well as kickoff/penalties repetitions, VAR interruptions and event description
    model, log_incidence = set_score(model, ball_data, params, log_incidence)  

    ## Add ball control to differentiate from `event_name`
    model.insert(7,'ball_control','')
    model = model.apply(set_event_ballcontrol, axis=1)

    ## Set dead ball event logic
    if params.get('set_event_logic',False):
        model = set_event_logic(model, params)

    ## Insert Duels
    model = set_duels(model, params)
    
    ## Insert event Ids
    t1 = time()
    model['time_ms'] = model['event_time'].apply(lambda x: x.to_event_time()[1])
    model['event_id'] = model['frame'].apply(lambda x: str(x) + '_' + params['segment_id'])

    ## Set team possession for every frame
    team_possession = set_team_possession(model, params)
    model = model.merge(team_possession[['frame','attacking_team_id']].rename(columns={'attacking_team_id':'team_possession'}), on='frame', how='inner')

    t2 = time()
    if params.get('set_possession_logic',False):
        ## Establish possession logic and chaining of events
        model, log_incidence = set_possession_logic(model, ball_data, params, log_incidence)
    t3 = time()

    if params.get('postprocess',False):
        ## Add contextual attributes to events
        model = postprocess(model, params, ball_data)
    t4 = time()

    ############################################ Cleanup ############################################
    model['jersey_team'] = model['player_id'].apply(jersey_team, args=(params,))
    model['contest_jersey_team'] = model['contest_player_id'].apply(jersey_team, args=(params,))

    model['name'] = model['player_id'].apply(names, args=(params,))
    model['contest_name'] = model['contest_player_id'].apply(names, args=(params,))

    model['player_role'] = model['player_id'].apply(roles, args=(params,))
    model['contest_player_role'] = model['contest_player_id'].apply(roles, args=(params,))
    
    try:
        model['tournament_stage'] =  params['tournament_stage']
    except:
        pass
            

    model['x'] = model['xy'].apply(unroll_tuple, args=(0,))
    model['y'] = model['xy'].apply(unroll_tuple, args=(1,))
    model['contest_x'] = model['contest_xy'].apply(unroll_tuple, args=(0,))
    model['contest_y'] = model['contest_xy'].apply(unroll_tuple, args=(1,))
    model['ball_x'] = model['ball_pos'].apply(lambda x: x[0])
    model['ball_y'] = model['ball_pos'].apply(lambda x: x[1])
    model['ball_z'] = model['ball_pos'].apply(lambda x: x[2])
    model.rename(columns={'ball_speed':'ball_speed_mps'}, inplace=True)

    if params.get('postprocess',False):
        model = model[['frame','event_id','event_time','time_ms',
                     'ball_control','event_name','event_outcome','contest','from_set_piece','dead_ball_event',
                     'possession_outcome','jersey_team','name','player_role',
                     'description','home_score','away_score','assist_event_id',
                     'previous_loss_event_id','next_gain_event_id','next_loss_event_id',
                     'incoming_angle','outgoing_angle','player_location','ball_from','packing_possession',
                     'packing_pass','distance_traveled_recipient',
                     'distance_traveled_ball_x','distance_traveled_ball_y',
                     'home_span_x','home_span_y','away_span_x','away_span_y',
                     'ball_x','ball_y','ball_z','ball_speed_mps','half','x','y','player_id','team_id','distance',
                     'contest_jersey_team','contest_name','contest_player_role',
                     'contest_x','contest_y','contest_player_id','contest_team_id','contest_distance',
                     'is_gain','is_goalkeeper',
                     'ball_displacement','change_in_direction','change_in_speed',
                     'step_change','loss_gain',
                     'team_possession']]
    else:
        if params.get('set_possession_logic',False):
            model = model[['frame','event_id','event_time','time_ms',
                         'ball_control','event_name','event_outcome','contest','from_set_piece','dead_ball_event',
                         'possession_outcome','jersey_team','name','player_role',
                         'description','home_score','away_score','assist_event_id',
                         'previous_loss_event_id','next_gain_event_id','next_loss_event_id',                     
                         'ball_x','ball_y','ball_z','ball_speed_mps','half','x','y','player_id','team_id','distance',
                         'contest_jersey_team','contest_name','contest_player_role',                         
                         'contest_x','contest_y','contest_player_id','contest_team_id','contest_distance',
                         'is_gain','is_goalkeeper',
                         'ball_displacement','change_in_direction','change_in_speed',
                         'step_change','loss_gain',
                         'team_possession']]
        else:
            model = model[['frame','event_id','event_time','time_ms',
                     'ball_control','event_name','event_outcome','contest','from_set_piece','dead_ball_event',
                     'jersey_team','name','player_role',
                     'description','home_score','away_score','assist_event_id',                   
                     'ball_x','ball_y','ball_z','ball_speed_mps','half','x','y','player_id','team_id','distance',
                     'contest_jersey_team','contest_name','contest_player_role',                     
                     'contest_x','contest_y','contest_player_id','contest_team_id','contest_distance',
                     'is_goalkeeper','ball_displacement','change_in_direction','change_in_speed',
                     'step_change','loss_gain',
                     'team_possession']]

    ## Create the return dictionary
    data_output = {}
    data_output['params'] = params
    data_output['models'] = {}
    data_output['log_incidence'] = log_incidence
    data_output['models']['db_intervals'] = db_intervals
    ########################################################################################

    ## Set individual and team pressures
    t5 = time()    
    if params['compute_pressures']:
        model, team_possession, team_pressures = set_pressure(model, team_possession, params)
        data_output['models']['team_pressures'] = team_pressures
    t6 = time()
    
    ## Set defensive modes
    if params['compute_modes']:
        def_modes = set_defensive_modes(model, team_possession, params)
        data_output['models']['defensive_modes'] = def_modes
        model = model.merge(def_modes[['frame','defensive_mode']], on='frame', how='inner')

    ## Extract pitch control information for passes
    t7 = time()
    if params['compute_pitch_control']:
        model = apply_pitch_control(model, params)
    t8 = time()
    
    data_output['models']['events'] = model
    data_output['models']['team_possession'] = team_possession    
    data_output['runtime'] = {'events':t1-t0, 'team_possession':t2-t1, 'temporal_logic':t3-t2, 'postprocess':t4-t3, 'cleanup': t5-t4,
                        'pressures':t6-t5, 'def_modes':t7-t6, 'pitch_control':t8-t7}

    return data_output


def set_duels(model, params):
    """
    This routine detects the duels as follows: If there is a pass/shot/cross while the possession is contested and the two players involved are
    within pz_contest, then we assign an air/ground duel based on the height of the ball. Additionally, we can check the frames 
    where a possession loss was originally recorded in `possession_model` but later overriden as a part of a 
    continued contested possession in `set_gains_losses`.

    :param model: Events dataframe
    :param params: Dictionary containing all the relevant parameters    
    :return: Events dataframe
    """ 
    
    hyperparams = params['hyperparams']
    duels = model[pd.eval("model.event_name.str.contains('|'.join(['pass','cross','shot','own'])) and model.contest != ''")]
    
    for row in duels.itertuples():
        player_1 = eval(row.contest_xy.split(';')[0])
        player_2 = eval(row.contest_xy.split(';')[1])
        ball_distance = row.contest_distance.split(';')
        player_distance = np.sqrt((player_1[0]-player_2[0])**2 + (player_1[1]-player_2[1])**2)
        
        if player_distance < hyperparams['pz_contest']:

            if len(model.at[row.Index, 'ball_pos']) == 3:
                model.at[row.Index,'contest'] = 'air duel' if model.at[row.Index, 'ball_pos'][2] >  params['aerial_z'] else 'ground duel'
            else:
                model.at[row.Index,'contest'] = 'duel'

    contest = model[pd.eval("model.contest != ''")].copy()
    contest['jump_frame'] = contest['frame'].diff()
    contest['jump_frame'].fillna(1,inplace=True)
    change_in_frame = contest.query("jump_frame > 1").index.tolist()

    ## Instances where there was originally a possession loss but later was erased because there was a continuation 
    ## players in possession. This can still be a duel since the ball distanced from the ball handler

    i = 0
    for frame in change_in_frame:
        this_contest = contest.loc[i:frame-1]
        i = frame
        if this_contest['contest'].str.contains('duel').any():
            continue
        if this_contest['step_change'].str.contains('18').any():
            continue
        if not this_contest['loss_gain'].str.contains('loss').any():
            continue

        j = 0
        for row in this_contest[this_contest['loss_gain'].str.contains('loss')].itertuples():            
            player_1 = eval(row.contest_xy.split(';')[0])
            player_2 = eval(row.contest_xy.split(';')[1])
            if j == 0:
                players_duel = [player_1, player_2]

            ball_distance = row.contest_distance.split(';')
            player_distance = np.sqrt((player_1[0]-player_2[0])**2 + (player_1[1]-player_2[1])**2)
            duel_annotated = True if j > 0 and player_1 in players_duel and player_2 in players_duel else False

            if player_distance < hyperparams['pz_contest']/2 and not duel_annotated:
                model.at[row.Index,'step_change'] += '-17'
                if len(model.at[row.Index, 'ball_pos']) == 3:
                    model.at[row.Index,'contest'] = 'air duel' if model.at[row.Index, 'ball_pos'][2] >  params['aerial_z'] else 'ground duel'
                else:
                    model.at[row.Index,'contest'] = 'duel'


    return model


def set_saves_shots(model, ball_dead, goalie_frames, ball_data, params):

    """
    This routine creates the logic to establish goalkeeper interventions as saves and the preceding possession losses as shots, 
    in addition to crosses. The hierarchy of cases is thoroughly detailed in the auto-eventing paper doi.org/10.1007/s12283-022-00381-6  
    Specific parts of the routine are commented for better explainability.     


    :param model: Events dataframe
    :param ball_dead: Dataframe of all frames where ball is dead
    :param goalie_frames: Frames where the goalkeeper first intervenes in a possession, as extracted from gains_losses
    :param ball_data: Dataframe of ball data    
    :param params: Dictionary containing all the relevant parameters    
    :return: Events dataframe updated
    """ 

    players_df = params['players_df']
    i_passwithsave = np.nan
    saves = model.query("frame.isin(@goalie_frames) or (step_change.isin(['a','b','c','d']) and is_goalkeeper==True)").reset_index().rename(columns={'index':'save_index'})
    players_in_areas = count_player_in_penalty_box(params)

    for row in saves.itertuples():
        i0 = row.save_index

        players_involved = row.player_id.split(';')
        teams_involved = row.team_id.split(';')
        goalie_id = [player for player in players_involved if player in params['goalkeeper_ids']][0]
        goalie_pos = eval(row.xy.split(';')[players_involved.index(goalie_id)])
        
        goalie_team = players_df.at[goalie_id,'tracking_team_id']

        try:
            i_lastpass = (model.loc[:i0-1,'event_name'].str.contains("|".join(['pass','shot','own'])))[::-1].idxmax()
        except:
            continue        
        
        if row.Index > 0 and i_lastpass == i_passwithsave:
            continue
        else:
            i_passwithsave = i_lastpass

        ## Check team of the player making the preceding possession loss
        if len(model.at[i_lastpass,'player_id'].split(';')) > 1:
            diff_team = True if len(set(model.at[i_lastpass,'team_id'].split(';'))) > 1 else model.at[i_lastpass,'team_id'].split(';')[0] != goalie_team

        else:
            diff_team = model.at[i_lastpass,'team_id'].split(';')[0] != goalie_team

        shooter_pos = eval(model.at[i_lastpass,'xy'].split(';')[0])
        
        ## Distance and zone of shot
        dist_shot = np.sqrt((shooter_pos[0]-goalie_pos[0])**2 + (shooter_pos[1]-goalie_pos[1])**2)
        zone = pitch_zone(shooter_pos, -params['active_goal'][goalie_team][0], params)
    
        ## If is a possession/contest WITHOUT a pass between shot and save, then it cannot be a shot/save pair
        other_player_between = (model.loc[i_lastpass+1:i0-1,'event_name'] != 'no control').any()
        if other_player_between:
            continue
        
        ## Determine if goalkeeper and shooter are inside the penalty box
        goalie_in_area = in_penalty_box(goalie_pos[0], goalie_pos[1],
                                        -np.sign(params['active_goal'][goalie_team][0]),
                                        params, tol=params["in_area"])

        shooter_in_area = in_penalty_box(shooter_pos[0], shooter_pos[1],
                                        -np.sign(params['active_goal'][goalie_team][0]),
                                        params)

        ## If the goalkeeper is outside the area or the preceding loss comes from a teammate, it is not a shot-save pair
        if not goalie_in_area or not diff_team:
            continue
            

        ## Logic to determine if the preceding possession loss was a cross
        ball_has_not_been_dead = (model.loc[i_lastpass:i0-1,'dead_ball_event'] == '').all()        

        ball_pos_pass = model.at[i_lastpass,'ball_pos']
        f_lastpass = model.at[i_lastpass,'frame']
        if ball_has_not_been_dead:
            ball_pos_goal = model.at[i0,'ball_pos']
            f0 = model.at[i0,'frame']
        else:
            i_dbe = (model.loc[i_lastpass:i0-1,'dead_ball_event'] != '').idxmax()
            ball_pos_goal = model.at[i_dbe,'ball_pos']
            f0 = model.at[i_dbe,'frame']
        
        ball_longer_than_distance = np.sqrt((ball_pos_goal[0]-ball_pos_pass[0])**2 + (ball_pos_goal[1]-ball_pos_pass[1])**2) >= params.get('cross_distance',0)
        if 'ball_z' not in ball_data.columns:
            ball_higher_than_height = True
        else:
            ball_higher_than_height = (ball_data.loc[f_lastpass:f0,'ball_z'] > params.get('cross_height',0)).any()
        
        try:
            opponents_in_area = players_in_areas[f'{goalie_team}_penbox'].at[row.frame,'count'] > 0
        except:
            opponents_in_area = 0        

        ball_dir = ball_direction(ball_pos_pass, ball_pos_goal)
        if shooter_pos[1]*ball_dir[1] < 0:
            towards_penalty_area = True
        else:
            towards_penalty_area = False
        
        is_cross = towards_penalty_area and ball_higher_than_height and ball_longer_than_distance

        ## Logic to determine if the preceding possession shot went forward and on target
        shot_forward, on_target_direction = shot_on_target(ball_pos_pass[0], ball_pos_pass[1], ball_dir, np.sign(goalie_pos[0]), params)
                
        ##  Figure out if there is a dead ball event after the GK intervention 
        i_pzbeforeoob = (model.loc[i0:,'dead_ball_event']!='').idxmax()
        all_frames_goalie = model.loc[i0:i_pzbeforeoob].query("event_name!='no control'")['player_id'].apply(match_player_interval,args=(goalie_id,)).all()
        goal_kick_after = (model.loc[i0:i_pzbeforeoob,'dead_ball_event'] == 'out for goal kick').any() and all_frames_goalie
        corner_kick_after = (model.loc[i0:i_pzbeforeoob,'dead_ball_event'] == 'out for corner kick').any() and all_frames_goalie
        goal_after = (model.loc[i0:i_pzbeforeoob,'dead_ball_event']== 'goal').any() and all_frames_goalie
        
        ## Figure out if there is any pass after the GK making contact with the ball, towards detecting deflections
        shot_from = model.at[i_lastpass,'from_set_piece']
        frames_save_deflect = int(params.get('save_deflect_interval',0)*params['fps'])
        interval_deflect = model.loc[i0:i0+frames_save_deflect]
        pass_after_save = (interval_deflect['event_name'].str.contains('pass') | ~interval_deflect['player_id'].str.contains(goalie_id)).any()
        i_passaftersave = (interval_deflect['event_name'].str.contains('pass') | ~interval_deflect['player_id'].str.contains(goalie_id)).idxmax() if pass_after_save else np.nan
                

        ## If there is a corner after, we signal goalie intervention as save, despite coming from same team (most likely data error)
        if goalie_in_area and not diff_team and corner_kick_after:            
            model.at[i_lastpass,'event_outcome'] = 'dead ball'
            model.loc[i0:i_pzbeforeoob-1,'dead_ball_event'] = ''
            model.loc[i_pzbeforeoob,'dead_ball_event'] = 'out for corner kick'
            model.loc[i0:i_pzbeforeoob-1,'step_change'] += '-10'
            model.loc[i_pzbeforeoob,'step_change'] += '-15'
            model.loc[i0,'event_name'] = 'save and deflect'
            model.loc[i0+1:i_pzbeforeoob,'event_name'] = 'no control'
            model.loc[i0:i_pzbeforeoob-1,'dead_ball_event'] = ''
            continue            
    
        ## In this decision tree we take care of all the options based on the hierarchy described in doi.org/10.1007/s12283-022-00381-6 
        if not ball_has_not_been_dead:
            ev_name = 'cross' if zone =='cross' and is_cross else ('shot not on target' if zone == 'shot' or shooter_in_area else 'pass')
            model.at[i_lastpass,'event_outcome'] = 'dead ball'
            
        elif ball_has_not_been_dead and goal_kick_after:
            model.loc[i0:i_pzbeforeoob-1,'dead_ball_event'] = ''
            model.loc[i_pzbeforeoob,'dead_ball_event'] = 'out for goal kick'
            model.loc[i0:i_pzbeforeoob-1,'step_change'] += '-10'
            model.loc[i_pzbeforeoob,'step_change'] += '-15'
            model.loc[i0:i_pzbeforeoob,'event_name'] = 'no control'
            ev_name = 'cross' if zone =='cross' and is_cross else ('shot not on target' if zone == 'shot' or shooter_in_area else 'pass')
            model.at[i_lastpass,'event_outcome'] = 'dead ball'

        elif ball_has_not_been_dead and corner_kick_after:
            model.loc[i0:i_pzbeforeoob-1,'dead_ball_event'] = ''
            model.loc[i_pzbeforeoob,'dead_ball_event'] = 'out for corner kick'
            model.loc[i0:i_pzbeforeoob-1,'step_change'] += '-10'
            model.loc[i_pzbeforeoob,'step_change'] += '-15'
            model.loc[i0,'event_name'] = 'save and deflect'
            model.loc[i0+1:i_pzbeforeoob,'event_name'] = 'no control'
            ev_name = 'cross' if zone =='cross' and is_cross else 'shot'
            if ev_name == 'shot' and on_target_direction:
                ev_name = 'shot on target'
            elif ev_name == 'shot' and not on_target_direction:
                ev_name = 'shot not on target'    
            model.at[i_lastpass,'event_outcome'] = 'saved'

        elif ball_has_not_been_dead and goal_after:
            
            model.loc[i0:i_pzbeforeoob-1,'dead_ball_event'] = ''
            model.loc[i_pzbeforeoob,'dead_ball_event'] = 'goal'
            model.loc[i0:i_pzbeforeoob-1,'step_change'] += '-10'
            model.loc[i_pzbeforeoob,'step_change'] += '-15'
            model.loc[i0+1:i_pzbeforeoob,'event_name'] = 'no control'
            model.loc[i0:i_pzbeforeoob,'event_outcome'] = ''
            if model.at[i0,'event_name'] != 'no control':
                model.loc[i0,'event_name'] = 'unsuccessful save'
            else:
                model.loc[i0,'event_name'] = 'no control'
                
            ev_name = 'shot on target'   
            model.at[i_lastpass,'event_outcome'] = 'goal'
            
        elif model.at[i0,'event_name'] != 'no control':

            if zone == 'cross' and is_cross:
                if opponents_in_area:
                    ev_name = 'cross'
                    save_name = 'claim and '
                    ev_outcome = 'claimed'
                else:
                    ev_name = 'pass'
                    save_name = 'reception from loose ball' if 'reception' in model.at[i0,'event_name'] else 'reception'
                    ev_outcome = 'loose ball'

            elif zone == 'shot' or shooter_in_area:
                if shot_forward and on_target_direction:
                    ev_name = 'shot on target'
                    save_name = 'save and '
                    ev_outcome = 'saved'
                elif shot_forward and not on_target_direction:
                    ev_name = 'shot not on target'
                    save_name = 'save and '
                    ev_outcome = 'saved'
                else:
                    ev_name = 'pass'
                    save_name = 'reception from loose ball' if 'reception' in model.at[i0,'event_name'] else 'reception'
                    ev_outcome = 'loose ball'                
            else:
                ev_name = 'pass'
                save_name = 'reception from loose ball' if 'reception' in model.at[i0,'event_name'] else 'reception'
                ev_outcome = 'loose ball'
                
            if pass_after_save and 'reception' not in save_name:
                model.loc[i0,'step_change'] += '-10'
                i = i0+1                
                while True:
                    try:
                        if (model.at[i,'is_goalkeeper'] == False or 
                            (model.at[i,'contest'] == 'contest' and model.at[i,'contest_player_id'].split(';')[0] != goalie_id) or 
                            i > i_passaftersave):
                            break
                        else:
                            model.loc[i,'event_name'] = 'no control'
                            model.loc[i,'step_change'] += '-10'
                            i += 1
                    except:
                        break
                del i
                save_outcome = 'deflect'          
            elif not pass_after_save and 'reception' not in save_name:
                save_outcome = 'retain'
                model.loc[i0,'step_change'] += '-10'
            else:
                save_outcome = ''
                model.loc[i0,'step_change'] += '-10'
                
            model.at[i0,'event_name'] = save_name + save_outcome
            model.at[i_lastpass,'event_outcome'] = ev_outcome
        else:
            goal_kick = (model.loc[i_lastpass:i0,'dead_ball_event'] == 'out for goal kick').any()
            if zone == 'cross' and is_cross:
                ev_name = 'cross'
            elif zone == 'shot' or shooter_in_area:
                if goal_kick or not on_target_direction:
                    ev_name = 'shot not on target'
                else:
                    ev_name = 'shot on target'
            else:
                ev_name = 'pass'
            
        model.loc[i_lastpass,'event_name'] = ev_name

    ## Fix own goals by goalkeeper, this is a heuristic to correct data errors
    owngoals = model.query("event_name == 'own goal'").reset_index().rename(columns={'index':'own_index'})
    for row in owngoals.itertuples():
        if row.is_goalkeeper:
            ## Check previous pass and set own goal iff it comes from a goalkeeper's teammate
            ## Note that this is for instances where the own goal is attributed to the goalkeeper ONLY   
            i_lastpass = (model.loc[:row.own_index-1,'loss_gain'].str.contains('loss'))[::-1].idxmax()
            shot_from_teammate = len(set(model.at[i_lastpass,'team_id'].split(';'))) == 1 and model.at[i_lastpass,'team_id'].split(';')[0] == model.at[row.own_index,'team_id']
            model.at[i_lastpass,'event_name'] = 'own goal' if shot_from_teammate else 'shot on target'
            model.at[i_lastpass,'step_change'] += '-9'
            model.at[i_lastpass,'event_outcome'] = 'goal'

            model.loc[i_lastpass+1:row.own_index-1,'event_name'] = model.loc[i_lastpass+1:row.own_index-1].apply(lambda x: 'no control' if x['is_goalkeeper'] == True and len(set(x['team_id'].split(';'))) == 1 else x['event_name'], axis=1)
            model.loc[i_lastpass+1:row.own_index-1,'step_change'] = model.loc[i_lastpass+1:row.own_index-1].apply(lambda x: x['step_change']+'-4' if x['is_goalkeeper'] == True and len(set(x['team_id'].split(';'))) == 1 else x['step_change'], axis=1)
            model.loc[i_lastpass+1:row.own_index-1,'event_outcome'] = ''
            model.at[row.own_index, 'event_name'] = 'unsuccessful save'
            model.at[row.own_index, 'event_outcome'] = ''
            model.at[row.own_index, 'step_change'] += '-9'


    ## Change order of attackers in contested shots such that the first player is the one attacking (despite being furthest from the ball)
    shots = model.query("event_name.str.contains('shot')")    
    for row in shots.itertuples():
        team_ids = row.team_id.split(';')

        if len(set(team_ids)) == 1:
            continue

        x = eval(row.xy.split(';')[0])[0]
        
        if x >= 0:
            attacks = ['attack' if params['active_goal'][t][0] > 0 else 'defend' for t in team_ids]
        else:
            attacks = ['attack' if params['active_goal'][t][0] < 0 else 'defend' for t in team_ids]
            
        if attacks[0] == 'defend':
            index_first = attacks.index('attack')
            model.at[row.Index, 'step_change'] += '-12'
            for col in ['player_id','team_id','xy','distance']:
                tm = getattr(row, col).split(';')
                model.at[row.Index, col] = ';'.join([tm[index_first]] + tm[:index_first] + tm[index_first+1:])

    return model



def set_possession_logic(model, ball_data, params, log_incidence):

    """
    This routine creates the temporal logic of possessions, namely the previous and next events for each pass and possession. 
    Additionally, we create the outcome of passes and possessions, as well as add crosses. 
    Specific parts of the routine are commented for better explainability.     

    :param model: Events dataframe
    :param ball_data: Dataframe of ball data    
    :param params: Dictionary containing all the relevant parameters    
    :param log_incidence: Dictionary where the incidences when running the possession model are logged    
    :return: Events dataframe, log_incidence updated
    """ 

    model['next_gain_frame'] = ''
    model['previous_loss_frame'] = ''
    model['next_loss_frame'] = ''
    model['possession_outcome'] = ''
    model['is_gain'] = False
    
    home_id = params['home_id']
    away_id = params['away_id']
    half = model.iloc[0]['half']
    segment_id = params['segment_id']
    passes = model[pd.eval("model.event_name.str.contains('|'.join(['pass','cross','shot','own','deflect']))")].reset_index().rename(columns={'index':'pass_index'})
    events = model[pd.eval("model.event_name.str.contains('|'.join(['pass','shot','reception','cross','save','claim','own']))")].reset_index().rename(columns={'index':'event_index'})
    dead_ball = model[pd.eval("model.dead_ball_event != ''")].reset_index().rename(columns={'index':'dbe_index'})  
    players_in_areas = count_player_in_penalty_box(params)

    ## First we establish the outcome of passes, as well as for the possession gain that follows the pass. This allows us to
    ## create a temporal event logic via event chaining
    pass_indexes = passes['pass_index'].tolist()
    event_indexes = events['event_index'].tolist()
    dbe_indexes = dead_ball['dbe_index'].tolist()
    for index0, pass_index in enumerate(pass_indexes):
        
        outcome = model.at[pass_index, 'event_outcome']
        try:
            next_events_index = [event_index for event_index in event_indexes if event_index > pass_index and event_index <= pass_index[index0+1]]
        except:
            next_events_index = [event_index for event_index in event_indexes if event_index > pass_index]
        
        if len(next_events_index) == 0:
            
            future_dbes = [dbe_index for dbe_index in dbe_indexes if pass_index <= dbe_index]
            if len(future_dbes) > 0:
                if model.at[future_dbes[0],'dead_ball_event'] == 'goal':
                    model.at[pass_index, 'event_outcome'] = model.at[future_dbes[0],'dead_ball_event'] if outcome == '' else outcome
                else:
                    model.at[pass_index, 'event_outcome'] = 'dead ball' if outcome == '' else outcome
            else:
                model.at[pass_index, 'event_outcome'] = 'end of half'
            continue
                
        future_dbes = [dbe_index for dbe_index in dbe_indexes if pass_index <= dbe_index and next_events_index[0] > dbe_index]
        if len(future_dbes) > 0:
            if model.at[future_dbes[0],'dead_ball_event'] == 'goal':
                model.at[pass_index, 'event_outcome'] = model.at[future_dbes[0],'dead_ball_event'] if outcome == '' else outcome
            else:
                model.at[pass_index, 'event_outcome'] = 'dead ball' if outcome == '' else outcome
            
            continue
        
        ## Establish the outcome of passes
        gain_index = next_events_index[0]
        next_event = model.loc[gain_index]
        row = model.loc[pass_index]
        model.at[pass_index,'next_gain_frame'] = str(next_event['frame'])
        model.at[gain_index,'previous_loss_frame'] = str(row.frame)
        

        if row.team_id.split(';')[0] != next_event.team_id.split(';')[0] and row.team_id.split(';')[0] == next_event.team_possession:
            ## Player for another team touches next ball but possession is not lost
            model.at[pass_index,'event_outcome'] = 'intercepted without control' if outcome == '' else outcome
            model.at[next_events_index[0],'event_name'] = 'uncontrolled interception' if next_event.event_name == 'reception' else next_event.event_name

        elif row.team_id.split(';')[0] != next_event.team_id.split(';')[0] and row.team_id.split(';')[0] != next_event.team_possession:
            ## Player for another team touches next ball and possession is lost
            model.at[pass_index,'event_outcome'] = 'intercepted' if outcome == '' else outcome
            model.at[next_events_index[0],'event_name'] = 'interception' if next_event.event_name == 'reception' else next_event.event_name
        else:
            model.at[pass_index,'event_outcome'] = 'completed' if outcome == '' else outcome
                
        ## Thus far, only passes/shots that involve a goalkeeper may be a cross. In here, we investigate if the rest can be a cross.
        if row.event_name == 'cross':
            continue
        
        pos = eval(row.xy.split(';')[0])
        
        next_pos = eval(next_event.xy.split(';')[0])
        
        ball_dir = ball_direction(row.ball_pos, next_event.ball_pos)
        
        if pos[1]*ball_dir[1] < 0:
            towards_penalty_area = True
        else:
            towards_penalty_area = False
        
        ball_longer_than_distance = np.sqrt((row.ball_pos[0]-next_event.ball_pos[0])**2 + (row.ball_pos[1]-next_event.ball_pos[1])**2) >= params['cross_distance']
        if 'ball_z' not in ball_data.columns or np.isnan(ball_data.loc[row.frame:next_event.frame,'ball_z']).any():
            ball_higher_than_height = True
        else:
            ball_higher_than_height = (ball_data.loc[row.frame:next_event.frame,'ball_z'] > params['cross_height']).any()

        is_cross = towards_penalty_area and ball_higher_than_height and ball_longer_than_distance
        

        pass_team_id = row.team_id.split(';')[0]

        zone = pitch_zone(pos, params['active_goal'][pass_team_id][0], params)
        defending_team_id = home_id if pass_team_id == away_id else away_id
        attacking_team_id = home_id if pass_team_id == home_id else away_id            

                        
        try:
            opponents_area = players_in_areas[f'{defending_team_id}_penbox'].at[next_event['frame'],'count'] > 0
        except:
            ## if the frame is not in players_in_areas, then it means ball was dead
            opponents_area = False
            
        next_event_in_cross_box = in_cross_box(next_pos[0], next_pos[1],
                                                      np.sign(params['active_goal'][attacking_team_id][0]),
                                                      np.sign(pos[1]), params)
        
        if zone == 'cross' and is_cross and next_event_in_cross_box:
            model.at[pass_index,'event_name'] = 'cross'
            

    ## Second we establish the outcome of possessions, as well as the preceding/subsequent possession loss
    retain_events = ['reception','interception','reception from loose ball','save and retain','claim and retain']
    gain_events = retain_events + ['save and deflect','claim and deflect','uncontrolled interception']
    receptions = model[pd.eval("model.event_name.isin(gain_events) or (model.event_name.str.contains('|'.join(['pass','shot','cross','own'])) and model.step_change.str.contains('|'.join(['2','3'])))")].reset_index().rename(columns={'index':'reception_index'})
    reception_indexes = receptions['reception_index'].tolist()
    passes = model[pd.eval("model.event_name.str.contains('|'.join(['pass','shot','cross','deflect']))")].reset_index().rename(columns={'index':'pass_index'})
    pass_indexes = passes['pass_index'].tolist()
    for index0, reception_index in enumerate(reception_indexes):
        
        model.at[reception_index,'is_gain'] = True
        previous_events_index = [event_index for event_index in pass_indexes if event_index < reception_index]
        if len(previous_events_index) == 0:
            msg = f"No pass preceding this reception, CHECK {model.at[reception_index,'event_time']}"
            log_incidence[model.at[reception_index,'frame']] = msg
            if params['verbose']:
                logging.warning(msg)
            continue
        model.at[reception_index,'previous_loss_frame'] = str(model.at[previous_events_index[-1],'frame'])
                
        if model.at[reception_index,'event_name'] == 'reception' and model.at[reception_index,'is_goalkeeper'] == True and model.at[reception_index,'ball_pos'][2] > params['gk_z_claim']:
            ## Heuristics to rename goalkeeper receptions to claims
            frames_save_deflect = int(params.get('save_deflect_interval',0)*params['fps'])
            interval_deflect = model.loc[reception_index:reception_index+frames_save_deflect]
            goalie_id = list(set(model.at[reception_index,'player_id'].split(';')).intersection(set(params['goalkeeper_ids'])))[0]
            pass_after_save = (interval_deflect['event_name'].str.contains('pass') | ~interval_deflect['player_id'].str.contains(goalie_id)).any()            

            model.at[reception_index,'event_name'] = 'claim and deflect' if pass_after_save else 'claim and retain'
            model.at[reception_index,'step_change'] += '-16'

        if model.at[reception_index,'event_name'] not in retain_events + ['uncontrolled interception']:
            model.at[reception_index,'next_loss_frame'] = str(model.at[reception_index,'frame'])
            model.at[reception_index,'possession_outcome'] = 'pass' if 'deflect' not in model.at[reception_index,'event_name'] else 'deflect'
            
            next_events_index = [event_index for event_index in event_indexes if event_index > reception_index]
            if len(next_events_index) == 0:
                future_dbes = [dbe_index for dbe_index in dbe_indexes if reception_index <= dbe_index]
                if len(future_dbes) > 0:
                    if model.at[future_dbes[0],'dead_ball_event'] == 'goal':
                        model.at[reception_index, 'possession_outcome'] = model.at[future_dbes[0],'dead_ball_event']
                    else:
                        model.at[reception_index, 'possession_outcome'] = 'dead ball'
                continue
                
            future_dbes = [dbe_index for dbe_index in dbe_indexes if reception_index <= dbe_index and next_events_index[0] > dbe_index]
            if len(future_dbes) > 0:
                if model.at[future_dbes[0],'dead_ball_event'] == 'goal':
                    model.at[reception_index, 'possession_outcome'] = model.at[future_dbes[0],'dead_ball_event']
                else:
                    model.at[reception_index, 'possession_outcome'] = 'dead ball'
            continue

        next_events_index = [event_index for event_index in event_indexes if event_index > reception_index]

        if len(next_events_index) == 0:
            future_dbes = [dbe_index for dbe_index in dbe_indexes if reception_index <= dbe_index]
            if len(future_dbes) > 0:
                if model.at[future_dbes[0],'dead_ball_event'] == 'goal':
                    model.at[reception_index, 'possession_outcome'] = model.at[future_dbes[0],'dead_ball_event']
                else:
                    model.at[reception_index, 'possession_outcome'] = 'dead ball'
            else:
                model.at[reception_index, 'possession_outcome'] = 'end of half'
            continue

        future_dbes = [dbe_index for dbe_index in dbe_indexes if reception_index <= dbe_index and next_events_index[0] > dbe_index]
        if len(future_dbes) > 0:
            dbe_index = future_dbes[0]
            dbe_event = 'dead ball'
        else:
            dbe_index = np.inf
            dbe_event = ''

        loss_index = next_events_index[0]
        interval = model.loc[reception_index:loss_index].copy().query("ball_control != 'no control'")
        player_id = model.at[reception_index,'player_id'].split(';')[0]
        team_id = model.at[reception_index,'team_id'].split(';')[0]

        ## This subroutine takes care of contested possessions where the closest player to the ball keeps changing without a passing event
        if (~interval['player_id'].str.contains(player_id)).any():
            loss_index = (~interval['player_id'].str.contains(player_id)).idxmax()
            new_team_id  = model.at[loss_index,'team_possession']
            no_contest = model.at[loss_index,'contest'] == ''
            if loss_index <= dbe_index:
                model.at[reception_index, 'next_loss_frame'] = str(model.at[loss_index-1,'frame'])
                model.at[reception_index,'possession_outcome'] = 'lost' if new_team_id != team_id and no_contest else 'maintained'
            else:
                model.at[reception_index, 'next_loss_frame'] = str(model.at[dbe_index,'frame'])
                model.at[reception_index,'possession_outcome'] = dbe_event
                continue

            interval_loop = interval
            while True:
                
                gain_index = loss_index
                interval_loop = interval_loop.loc[gain_index:]                
                if interval_loop.empty:
                    break

                player_id = model.at[gain_index,'player_id'].split(';')[0]
                model.at[gain_index,'is_gain'] = True
                model.at[gain_index, 'previous_loss_frame'] = str(model.at[loss_index-1,'frame'])
                
                team_id = model.at[gain_index,'team_id'].split(';')[0]
                if (~interval_loop['player_id'].str.contains(player_id)).any():
                    loss_index = (~interval_loop['player_id'].str.contains(player_id)).idxmax()
                    new_team_id  = model.at[loss_index,'team_possession']
                    no_contest = model.at[loss_index,'contest'] == ''
                    if loss_index <= dbe_index:
                        model.at[gain_index, 'next_loss_frame'] = str(model.at[loss_index-1,'frame'])
                        model.at[gain_index,'possession_outcome'] = 'lost' if new_team_id != team_id and no_contest else 'maintained'
                    else:
                        model.at[gain_index, 'next_loss_frame'] = str(model.at[dbe_index,'frame'])
                        model.at[gain_index,'possession_outcome'] = dbe_event
                        break
                else:
                    loss_index = next_events_index[0]
                    if loss_index <= dbe_index:
                        model.at[gain_index, 'next_loss_frame'] = str(model.at[loss_index,'frame'])
                        model.at[gain_index,'possession_outcome'] = 'pass'
                    else:
                        model.at[gain_index, 'next_loss_frame'] = str(model.at[dbe_index,'frame'])
                        model.at[gain_index,'possession_outcome'] = dbe_event
                    break

        else:

            if loss_index <= dbe_index:
                model.at[reception_index, 'next_loss_frame'] = str(model.at[loss_index,'frame'])
                model.at[reception_index,'possession_outcome'] = 'pass'
            else:
                model.at[reception_index, 'next_loss_frame'] = str(model.at[dbe_index,'frame'])
                model.at[reception_index,'possession_outcome'] = dbe_event 
            
    ## Set Event ID of the preceding and succeeding possession loss and the succeeding possession gain, when applicable
    model['previous_loss_event_id'] = model['previous_loss_frame'].apply(lambda x: str(x) + '_' + segment_id if x!= '' else x)
    model['next_loss_event_id'] = model['next_loss_frame'].apply(lambda x: str(x) + '_' + segment_id if x!= '' else x)
    model['next_gain_event_id'] = model['next_gain_frame'].apply(lambda x: str(x) + '_' + segment_id if x!= '' else x)
    return model, log_incidence


def set_score(model, ball_data, params, log_incidence):
    """
    This routine updates the score from the events that have been detected. It also checks for invalid kickoffs/penalties that 
    have to be repeated, as well as goals that are awarded late by VAR. Finally, it adds a description column and information on assists.
    Specific parts of the routine are commented for better explainability. 


    :param model: Events dataframe
    :param ball_data: Dataframe of ball data    
    :param log_incidence: Dictionary where the incidences when running the possession model are logged    
    :param params: Dictionary containing all the relevant parameters    
    :return: Events dataframe, log_incidence updated
    """ 

    model['home_score'] = 0
    model['away_score'] = 0
    model['description'] = ''
    model['assist_event_id'] = ''
    
    penbox_len = params['pitch_measures']['penbox_len']

    ## Check that kickoff start was not repeated. If it is, set start of half to new kickoff
    index_kickoff = model.query("from_set_piece == 'kickoff'").index.tolist()
    index_kickoff_start = model.query("from_set_piece == 'kickoff start'").index.tolist()[0]

    if len(index_kickoff) > 0:
        ball_not_to_goal = model.loc[:index_kickoff[0],'ball_pos'].apply(lambda x: abs(x[0]) < params['pitch_length']/2 - penbox_len).all()
        if ball_not_to_goal:
            msg = f"Start kickoff had to be repeated at {model.at[index_kickoff[0],'event_time']}, resetting time to fit new kickoff"
            log_incidence[model.at[index_kickoff[0],'frame']] = msg
            if params['verbose']:
                print(msg)
            
            model.at[index_kickoff_start,'from_set_piece'] = 'incorrect kickoff'
            model.loc[:index_kickoff[0]-1,'event_name'] = 'no control'
            model.loc[:index_kickoff[0]-1,'event_outcome'] = ''
            model.loc[:index_kickoff[0]-1,'dead_ball_event'] = 'referee interruption'
            model.loc[:index_kickoff[0]-1,'step_change'] = '13'

            model.at[index_kickoff[0],'from_set_piece'] = 'kickoff start'
            
            lag = model.at[index_kickoff[0],'event_time'].to_event_time()[1]/1000
            model['event_time'] = model['event_time'].apply(lambda x: x.add_delta_seconds(-lag))

    ## Correct instances where the first contact with the kickoff taker is marked as reception, which should not be. It takes into account
    ## incorrect kickoffs as well, as everything that happened before that is not a reception
    index_kickoff_start = model.query("from_set_piece == 'kickoff start'").index.tolist()[0]
    if (model.loc[:index_kickoff_start,'event_name'] == 'reception').any():
        model.loc[np.where(model.loc[:index_kickoff_start,'event_name'] == 'reception')[0],'event_name'] = 'possession'


    ## Detect instances when a penalty is repeated
    index_penkick = model.query("from_set_piece == 'penalty kick'").index.tolist()
    goalkeeper_ids = params['goalkeeper_ids']
    if len(index_penkick) > 1:
        for i, i_pen in enumerate(index_penkick):
            if i > 0:
                ball_not_outside_area = model.loc[index_penkick[i-1]:i_pen,'ball_pos'].apply(lambda x: abs(x[0]) > params['pitch_length']/2 - penbox_len).all()
                if ball_not_outside_area:
                    i_wrong_pen = index_penkick[i-1]
                    i_ref_interruption = (model.loc[i_wrong_pen:i_pen,'dead_ball_event'] == 'penalty awarded').idxmax()
                    f_wrong_pen = model.at[i_wrong_pen,'frame']   
                    goalie_team_id = params['home_id'] if model.at[i_wrong_pen,'team_id'].split(';')[0] == params['away_id'] else params['away_id']                 
                    goalie_distance = abs(params['td'].query("frame == @f_wrong_pen and team_id == @goalie_team_id and player_id.isin(@goalkeeper_ids)", engine='python').iloc[0]['x'] - params['pitch_length']/2)
                    msg = f"There was a penalty kick recorded at {model.at[i_wrong_pen,'event_time']} that was repeated, goalkeeper {round(goalie_distance,2)}m from line"
                    log_incidence[model.at[i_wrong_pen,'frame']] = msg
                    if params['verbose']:
                        print(msg)
                    model.at[i_ref_interruption,'dead_ball_event'] = 'referee interruption'
                    model.loc[i_wrong_pen:i_ref_interruption,'event_name'] = 'no control'
                    model.loc[i_wrong_pen:i_ref_interruption,'event_outcome'] = ''
                    model.loc[i_wrong_pen:i_ref_interruption,'step_change'] += '-13'
                    model.at[i_wrong_pen,'from_set_piece'] = 'incorrect penalty'
                    

    ## Loop over goals to establish changes in score, assist information and correction of mistakes
    index_goal = model.query("dead_ball_event == 'goal'").index.tolist()  
    home_id = params['home_id']
    away_id = params['away_id']
    half = model.iloc[0]['half']
    segment_id = params['segment_id']
    home_active = params['active_goal'][home_id]
    away_active = params['active_goal'][away_id]
    home_score = 0
    away_score = 0 
    for i, i_goal in enumerate(index_goal): 
        ball_pos = model.at[i_goal, 'ball_pos']        
        
        ## If we have a sequence kickoff1-goal-kickoff1, but between kickoff1-goal the ball does arrive to the penalty areas,
        ## then it corresponds to a kickoff that has to be repeated and the goal is a false positive, we take care of this here
        if i > 0:
            prev_kickoff = (model.loc[index_goal[i-1]:,"from_set_piece"] =='kickoff').idxmax()
            ball_not_to_goal = model.loc[prev_kickoff:i_goal,'ball_pos'].apply(lambda x: abs(x[0]) < params['pitch_length']/2 - penbox_len).all()
            if ball_not_to_goal:
                msg = f"There was a goal recorded at {model.at[i_goal,'event_time']} without a shot after the previous goal, deleting"
                log_incidence[model.at[i_goal,'frame']] = msg
                if params['verbose']:
                    print(msg)
                model.at[i_goal,'dead_ball_event'] = 'referee interruption'
                model.at[i_goal,'event_name'] = 'pass' if model.at[i_goal,'event_name']!= 'no control' else 'no control'
                model.at[i_goal,'step_change'] += '-13'
                model.at[prev_kickoff,'from_set_piece'] = 'incorrect kickoff'
                continue
                            
            
        i_shot = model.loc[:i_goal,'event_name'].str.contains("|".join(['shot','own','cross']))[::-1].idxmax()    
        f_shot = model.at[i_shot,'frame']
        f_goal = model.at[i_goal,'frame']        
        scorer = model.at[i_shot,'player_id'].split(';')
        teams = model.at[i_shot,'team_id'].split(';')                        
            
        ## Determining the team that scored based on the team that kickoffs, or if there is no kickoff
        ## we use distance of ball to both goals at the time of shot.        
        dist_home_active = abs(home_active[0]-ball_pos[0])
        dist_away_active = abs(away_active[0]-ball_pos[0])        
        if (model.loc[i_goal:,"from_set_piece"] =='kickoff').any():
            next_kickoff = (model.loc[i_goal:,"from_set_piece"] =='kickoff').idxmax()
            team_received = model.at[next_kickoff,'team_id'].split(';')[0]            
        else:
            msg = f"Determining scoring team based on distance ball-goal {model.at[i_shot,'event_time']}"
            log_incidence[model.at[i_shot,'frame']] = msg
            if params['verbose']:
                print(msg)
            team_received = away_id if dist_home_active < dist_away_active else home_id

        team_scored = home_id if team_received == away_id else away_id
        active_goal = params['active_goal'][team_scored][0]
        ## Correct wrong `own goal` to `shot on target` if the scoring team is the same as the team of the player closest to ball at shot time
        isown = False
        if model.at[i_shot,'event_name'] == 'own goal' and teams[0] != team_received:
            model.at[i_shot,'event_name'] == 'shot on target'
            model.at[i_shot,'step_change'] += '-14'
        elif model.at[i_shot,'event_name'] == 'own goal' and teams[0] == team_received:    
            isown = True


        ## Update score
        if team_scored == home_id:
            home_score += 1
        else:
            away_score += 1
            

        ## If the ball has not reached the penalty box from the last shot to goal, most likely is a goal awarded after checking VAR, 
        ## and said shot was not a shot. Exception if shot has been imputed on a dead ball frame from a set piece. 
        ## We disambiguate by looking at the last pass before the goal where the ball goes forward, on target, and ball has reached the penalty box
        segment = ball_data.loc[f_shot:f_goal,['ball_x','ball_y']].apply(lambda x: in_penalty_box(x['ball_x'],x['ball_y'],
                                                                                                  np.sign(active_goal), params, 
                                                                                                  tol=params['goal_var']), axis=1)
        if ~segment.any() and not any([char in model.at[i_shot,'step_change'] for char in ['a','b','c']]):                 
            prev_passes = model.query("frame < @f_shot and event_name.str.contains('|'.join(['shot','own','pass','cross']))")[::-1]
            for prev_shot in prev_passes.itertuples():
   
                ball_x, ball_y, ball_dir = ball_data.loc[prev_shot.frame,['ball_x','ball_y','direction_1']]
                
                    
                shot_forward, on_target_direction = shot_on_target(ball_x, ball_y, ball_dir, active_goal, params)
                
                ball_in_penalty_box = in_penalty_box(ball_x, ball_y, np.sign(active_goal),params,tol=params['goal_var'])
                

                if any(substring in prev_shot.event_name for substring in ['shot','own','cross']) or (ball_in_penalty_box and 
                                                                                              shot_forward and on_target_direction):                    
                    
                    this_teams = model.at[i_shot,'team_id'].split(';')
                    i_prevshot = prev_shot.Index
                    model.at[i_prevshot,'event_name'] = 'shot on target' if team_scored in this_teams else 'own goal'
                    model.at[i_prevshot,'event_outcome'] = 'goal'
                    model.at[i_prevshot,'dead_ball_event'] = 'goal'                
                    model.loc[i_prevshot+1:i_goal,'event_name'] = 'no control'
                    model.loc[i_prevshot+1:i_goal,'dead_ball_event'] = ''
                    model.loc[i_prevshot+1:i_goal,'from_set_piece'] = ''
                    model.loc[i_prevshot+1:i_goal,'step_change'] += '-16'
                    model.at[i_goal,'description'] = f"VAR awards goal to {team_scored} scored at {model.at[i_prevshot,'event_time']}"
                    msg = f"Ball did not reach attacking box between {model.at[i_shot,'event_time']} and {model.at[i_goal,'event_time']}, goal given by VAR at {model.at[i_prevshot,'event_time']}"
                    log_incidence[model.at[i_shot,'frame']] = msg
                    if params['verbose']:
                        print(msg)                                    
                    
                    ## Change order of attackers in contested shots such that the first player is the one attacking
                    if len(set(prev_shot.team_id.split(';'))) > 1:
                        x = eval(prev_shot.xy.split(';')[0])[0]
                        team_ids = prev_shot.team_id.split(';')
                        if x >= 0:
                            attacks = ['attack' if params['active_goal'][t][0] > 0 else 'defend' for t in team_ids]
                        else:
                            attacks = ['attack' if params['active_goal'][t][0] < 0 else 'defend' for t in team_ids]

                        if attacks[0] == 'defend':
                            index_first = attacks.index('attack')
                            model.at[i_prevshot, 'step_change'] += '-12'
                            for col in ['player_id','team_id','xy','distance']:
                                tm = getattr(prev_shot, col).split(';')
                                model.at[i_prevshot, col] = ';'.join([tm[index_first]] + tm[:index_first] + tm[index_first+1:])    
                            
                    i_goal = i_prevshot    
                    break
                    
        ## Determining assists
        if model.at[i_shot,'from_set_piece'] != '' or isown:
            i_assist = None
        else:
            i_assist = model.loc[:i_goal,'event_name'].str.contains("|".join(['pass']))[::-1].idxmax()
            assister = model.at[i_assist,'player_id'].split(';')
             
        ## Description of goals       
        if len(set(teams)) == 1 or isown:               
            if teams[0] == team_received:
                msg = f"Should have been flagged as own goal at {model.at[i_shot,'event_time']}"
                log_incidence[model.at[i_goal,'frame']] = msg
                if params['verbose']:
                    print(msg)
                model.at[i_shot,'event_name'] = 'own goal' 
                model.at[i_shot,'step_change'] += '-14'
                isown = True
            player_scored = scorer[0]
            model.at[i_goal,'description'] = f"{'own ' if isown else ''}goal scored by:{player_scored},{team_scored}"                    
        else:# useful if two players from opposing teams are listed as shot takers, it will not be an own goal but a goal from the scoring team closest to ball
            player_scored = scorer[[t == team_scored for t in model.at[i_shot,'team_id'].split(';')].index(True)]
            model.at[i_goal,'description'] = f'goal scored by:{player_scored},{team_scored}'

        if i_assist is not None and home_id in model.at[i_assist,'team_id'].split(';'):
            team_assister = [t == home_id for t in model.at[i_assist,'team_id'].split(';')]
            player_assisted = assister[team_assister.index(True)]
            if player_scored != player_assisted:
                model.at[i_goal,'description'] += f', assisted by:{player_assisted},{team_scored}'
                model.at[i_goal,'assist_event_id'] = str(model.at[i_assist,'frame']) + '_' + segment_id
                
 
        model.loc[i_goal:, 'home_score'] = home_score
        model.loc[i_goal:, 'away_score'] = away_score
    
    
    ## Description of penalty kicks
    index_pks = model.query("from_set_piece == 'penalty kick'").index.tolist()  
    frames_goal_from_pk_lookahead = params['fps']*5
    for i, index_pk in enumerate(index_pks): 
        ball_pos = model.at[index_pk, 'ball_pos']
        this_frame = model.at[index_pk,'frame']
        lookahead_frame = this_frame + frames_goal_from_pk_lookahead
        dist_home_active = abs(home_active[0]-ball_pos[0])
        dist_away_active = abs(away_active[0]-ball_pos[0])
        team_kicked = home_id if dist_home_active < dist_away_active else away_id
        
        i_decision = (model.loc[:index_pk,'dead_ball_event'] == 'penalty awarded')[::-1].idxmax()
        
        next_goal = (model.query("frame >= @this_frame and frame <= @lookahead_frame")['dead_ball_event'] == 'goal')
        no_other_shot = (model.query("frame > @this_frame and frame <= @lookahead_frame")['event_name'] != 'shot on target').all()
        i_next = next_goal.idxmax() if next_goal.any() else np.nan
        

        if not np.isnan(i_next) and no_other_shot:
            outcome = 'scored'
            model.at[i_next,'description'] = ''


        elif not np.isnan(i_next) and not no_other_shot:
            outcome = 'missed'
            model.at[index_pk,'event_outcome'] = 'penalty missed'
            
        else:
            outcome = 'missed'
            model.at[index_pk,'event_outcome'] = 'penalty missed'
        
        shooter = model.at[index_pk,'player_id'].split(';')[0]
        if dist_home_active < dist_away_active:
            model.at[index_pk,'description'] = f'penalty {outcome} by:{shooter},{home_id}'
            model.at[i_decision,'description'] = f'penalty awarded to {home_id}'
        else:
            model.at[index_pk,'description'] = f'penalty {outcome} by:{shooter},{away_id}'
            model.at[i_decision,'description'] = f'penalty awarded to {away_id}'

    
    return model, log_incidence



def set_gains_losses(model, params, log_incidence):

    """
    This routine establishes a temporal logic for possession gains and losses, as well as continued possessions,
    building on the raw information extracted in possession_model.
    Specific parts of the routine are commented for better explainability. 

    :param model: Events dataframe
    :param params: Dictionary containing all the relevant parameters    
    :param log_incidence: Dictionary where the incidences when running the possession model are logged    
    :return: Events dataframe, frames where a goalie has possession and log_incidence updated.
    """ 

    hyperparams = params['hyperparams']

    ## First evaluate whether a possession loss should be labeled as a pass or not, depending on how the ball displacement compares to 
    ## the minimum displacement allowed. This is useful to avoid false positives where the remains ball still while the player is moving away.

    for row in model.itertuples():
        if 'loss' in row.event_name:
            if row.ball_displacement > hyperparams['min_ball_displacement']:
                event_name = 'pass'
            else:
                event_name = 'possession'
        else:
            event_name = row.event_name

        model.loc[row.Index, 'event_name'] = event_name



    ## Then, we care of continued possessions, namely a player is in possession, records a possession loss but remains in
    ## possession in the subsequent frames. This situation should not be a possession loss as the player may be simply
    ## in continued possession and their distance to the ball continuously increasing and decreasing as they run with
    ## the ball in close proximity. Assuming the ball has not been dead between successive possessions, we establish the following:
    ## If continued_possession = 0 we remove the pass if the closest player in this frame also has the ball within their PZ on the next frame
    ## If continued_possession = 1 we remove the pass if any of the players that have the ball within their PZ in this frame also do on the next frame
    ## The default should be continued_possession = 0 as it typically yields better results.

    passes = model[pd.eval("model.event_name.str.contains('pass')")].reset_index().rename(columns={'index':'pass_index'})
    for row in passes.itertuples():
        if row.Index == passes.shape[0]-1:
            break

        i0 = row.pass_index
        next_row = model.loc[i0+1]

        if hyperparams['continued_possession'] == 0:
            player_next_frame = row.player_id.split(';')[0] in next_row.player_id.split(';')
        elif hyperparams['continued_possession'] == 1:
            player_next_frame = match_player_interval(next_row.player_id, row.player_id)

        ball_not_been_dead = next_row.index_with_nan == row.index_with_nan + 1
        if player_next_frame and ball_not_been_dead:
            model.loc[i0,'step_change'] = '1'
            if row.event_name == 'pass':      
                model.loc[i0,'event_name'] = 'possession'


    ## We then shift to establishing gains. Logic for establishing whether player becomes in contact is governed by the hyperparameter
    ## continued_possession. If set to 1, player becomes in contact iff the ball was not in their PZ in the previous frame. If set to 0, player
    ## becomes in contact iff he was not the closest player to the ball in the previous frame. 
    ## The default should be continued_possession = 0 as it typically yields better results.
    for row in model.loc[1:].itertuples():
        prev_frame_players = model.at[row.Index-1,'player_id'].split(';')
        this_frame_players = row.player_id.split(';')

        if row.player_id != '':
            ball_has_been_dead = row.index_with_nan - model.at[row.Index-1,'index_with_nan'] > 1
            if hyperparams['continued_possession'] == 0:
                if prev_frame_players[0] not in this_frame_players or ball_has_been_dead:
                    model.loc[row.Index,'event_name'] = 'gain'
                    model.loc[row.Index,'step_change'] = '3'
            elif hyperparams['continued_possession'] == 1:
                if set(prev_frame_players).isdisjoint(this_frame_players) or ball_has_been_dead:
                    model.loc[row.Index,'event_name'] = 'gain'
                    model.loc[row.Index,'step_change'] = '3'


    goalie_frames = []
    gains = model.loc[model['event_name']=='gain'].reset_index().rename(columns={'index':'gain_index'})
    
    ## Set the gains/losses logic for each gain recorded above
    for row in gains.itertuples():
        i0 = row.gain_index
        ## Edge case for last event
        if row.Index == gains.shape[0]-1:
            i1 = model.shape[0]-1
        else:
            i1 = gains.at[row.Index+1,'gain_index'] - 1
      
        if i1 == i0:
          ## If possession only lasts for one frame, is either `no control` or a `pass`
            if not row.ball_changed_direction or not in_pitch_player(row.xy, params):
                ## If the ball did not change direction or the player is not in the pitch limits, it is no control
                model.loc[i0,'event_name'] = 'no control'
                model.loc[i0,'step_change'] = '4'
            else:
                ## Is only a pass if the ball has moved beyond minimum ball displacement hyperparameter, instead it is a `reception`.
                ## It is useful for instance when a foul is called after the player comes in contact
                if row.ball_displacement > hyperparams['min_ball_displacement']:
                    model.loc[i0,'step_change'] = '2'
                    model.loc[i0,'event_name'] = 'pass'
                    
                else:
                    model.loc[i0,'step_change'] = '3'
                    model.loc[i0,'event_name'] = 'reception'
                    
      
            if row.is_goalkeeper:
                ## Check if goalkeeper that receives is not on attacking side
                goalies_gain = [(player,row.team_id.split(';')[i], eval(row.xy.split(';')[i])[0]) for i,player in enumerate(row.player_id.split(';')) if player in params['goalkeeper_ids']]
                x_active = params['active_goal'][goalies_gain[0][1]][0]
                if not(len(goalies_gain) == 1 and abs(goalies_gain[0][2]-x_active) < abs(goalies_gain[0][2]+x_active)):
                    goalie_frames.append(row.frame)

        else:
            ## If possession lasts for more frames, we investigate whether the ball changes direction
            ## during the entire possession sequence or changes direction/speed at any individual frame

            interval = model.loc[i0:i1].copy()
      
            dir_0, dir_1 = interval.at[i0,'ball_dir_0'], interval.at[i1,'ball_dir_1']

            if dir_0 == (np.nan,np.nan) or dir_1 == (np.nan,np.nan):
                dir_change_interval = False
            else:
                dir_change_interval = (dir_0[0]*dir_1[0] + dir_0[1]*dir_1[1] < 
                                       hyperparams['ball_change_direction'] 
                                       if dir_1 != (0,0) else False)
        
            dir_change_frame = (interval['ball_changed_direction'] == True).any()
            vel_change_frame = (interval['ball_changed_speed'] == True).any()

            interval['in_pitch'] = interval['xy'].apply(lambda p: in_pitch_player(p, params))
            not_in_pitch = (interval['in_pitch'] == False).all()
            bool_2_frame = dir_change_interval or dir_change_frame or vel_change_frame
            if not bool_2_frame or not_in_pitch:
                ## No changes in direction/speed or the player was out of the pitch limits, we assign `no control`
                model.loc[i0:i1,'event_name'] = 'no control'
                model.loc[i0:i1,'step_change'] = '4'
            else:
                ## Changes in direction/speed, first frame is then a `reception`
                model.loc[i0,'step_change'] = '3'
                model.loc[i0,'event_name'] = 'reception'                   
                
                ## This is a heuristic to address situations where a continued contested possession occurs
                ## with the same players remaining near the ball, hence we should only detect a pass at the end of the interval
                ## To that end, we chunk the interval into sub-intervals, where sub-intervals consists of contiguous frames
                ## and we jump from one sub-interval to the next where the original possession model detected a loss, signaled
                ## by `step_change` == 1, but the routine above removed the loss as players were the same before/after the loss
                ## We omit the goalkeeper for these situations. 
                if (interval['event_name'].str.contains('pass')).any() and \
                  (interval['contest'].str.contains('contest')).any() and \
                   (interval['is_goalkeeper'] == False).all():

                    if (interval['step_change'] == '1').any():
                        ## If there has been an intermediate possession loss but the same players remain near the ball, we look at the very last
                        ## sub-interval within this continued possession
                        ii = interval.loc[interval['step_change'] == '1'].index[-1]
                        interval_sub = model.loc[ii+1:i1] 
                        if not ((interval_sub['ball_changed_direction'] == True).any() or
                                (interval_sub['ball_changed_speed'] == True).any() or 
                                (interval_sub['event_name'].str.contains('pass')).any()):
                            ## If there are no conditions for the last sub interval 
                            ## to accomodate a pass, we will look for the pass on the entire interval
                            interval_sub = interval
                    else:
                        ## There has not been an intermediate possession loss, hence we look for the pass on the entire interval
                        interval_sub = interval

                    ## On the sub-interval selected, we will assign a pass based on the following hierarchy
                    ## 1) Latest frame where ball changed direction
                    ## 2) Latest frame where ball changed speed
                    ## 3) Latest frame where a pass was annotated
                    ## We then clean up for the frames before/after the pass                    
                    if (interval_sub['ball_changed_direction'] == True).any():
                        i_pass = (interval_sub['ball_changed_direction'])[::-1].idxmax()
                    elif (interval_sub['ball_changed_speed'] == True).any():
                        i_pass = (interval_sub['ball_changed_speed'])[::-1].idxmax()
                    elif (interval_sub['event_name'].str.contains('pass')).any():
                        i_pass = (interval_sub['event_name'].str.contains('pass')).idxmax()
                    else:
                        msg = f"Check gain-loss at {model.at[i_0,'event_time']}"
                        log_incidence[model.at[i_0,'event_time']] = msg
                        if params['verbose']:
                            print(msg)
                    model.at[i_pass,'event_name'] = 'pass'
                    model.loc[i_pass+1:i1,'event_name'] = 'no control'
                    if i0+1 <= i_pass-1:
                        model.loc[i0+1:i_pass-1,'event_name'] = 'possession'
                
                

            if row.is_goalkeeper:
                ## Check if goalkeeper that receives is not on attacking side
                goalies_gain = [(player,row.team_id.split(';')[i], eval(row.xy.split(';')[i])[0]) for i,player in enumerate(row.player_id.split(';')) if player in params['goalkeeper_ids']]
                x_active = params['active_goal'][goalies_gain[0][1]][0]
                if not(len(goalies_gain) == 1 and abs(goalies_gain[0][2]-x_active) < abs(goalies_gain[0][2]+x_active)):
                    goalie_frames.append(row.frame)



    return model, goalie_frames, log_incidence
 



def apply_pitch_control(model, params):

    """
    This routine computes the pitch control gain and the passing lines for each frame where a player makes an uncontested pass. The passing lines
    are established as the straight lines between the passing player and the teammates that are within the player's cone of vision.
    Pitch control gain is defined as the difference between the PC on the location of the potential recipients of the pass and the origin of the pass
    Pitch control passing line is defined as a dictionary with the min, avg and max values of the pitch control along the passing line

    :param model: Events dataframe
    :param params: Dictionary containing all the relevant parameters        
    :return: Events dataframe updated with 'pc_gain','pc_passing_line','within_vision' columns   
    """ 

    df_poss = model.query("contest == '' and event_name.str.contains('|'.join(['pass','shot','cross'])) and team_id == team_possession and from_set_piece == ''").reset_index(drop=True).copy()
    for c in ['x','y','player_id','team_id','distance']:
        df_poss[c] = df_poss[c].apply(lambda x: x.split(";")[0] if ';' in x else x)

    df_poss['x'] = df_poss.x.astype(float)
    df_poss['y'] = df_poss.y.astype(float)
    df_poss['frame'] = df_poss.frame.astype(int)

    frame_player_poss_map = dict(zip(df_poss.frame, df_poss.player_id))
    frame_team_poss_map = dict(zip(df_poss.frame, df_poss.team_possession))

    pc_params = pc.default_model_params()

    ## Create the suitable player structure (position and velocity) to be fed to the PC model
    home_players = []
    away_players = []
    players_df = params['players_df']

    for p in players_df.itertuples():

        player_td = params['td'][pd.eval("params['td'].player_id == p.Index")].copy()
        player_td['xy'] = list(zip(player_td.x, player_td.y))
        player_td['speed'] = list(zip(player_td.vx, player_td.vy))
        pos_vel = player_td[['frame','xy','speed']].set_index('frame').to_dict(orient='index')

        ## create a player object for player_id 'p'
        if p.is_home == 1:
            home_players.append(pc.player(p, p.tracking_team_id,
                                       p.team_name,
                                       pc_params, p.is_goalkeeper,
                                       pos_vel))
        elif p.is_home == 0:
            away_players.append(pc.player(p, p.tracking_team_id,
                                       p.team_name,
                                       pc_params, p.is_goalkeeper,
                                       pos_vel))

    players_pc = {}
    players_pc['home'] = home_players
    players_pc['away'] = away_players



    ## For each frame where a player is passing the ball, evaluate for each teammate within the passer's vision cone the gain in PC 
    ## and the minimum, average and maximum of all the PC values evaluated every x metres along the passing line

    td = params['td'].merge(model[['frame','team_possession']], on='frame',how='inner')
    frame_groups = td[pd.eval("td.team_id == td.team_possession")][['frame','x','y','player_id','vx','vy']].reset_index(drop=True).groupby("frame")
    del td

    frame_adjacencies = {}
    df_poss['pc_gain'] = None
    df_poss['pc_passing_line'] = None
    df_poss['within_vision'] = None


    team_map = {params['home_id']:'home',
                params['away_id']:'away'}

    for row in df_poss.itertuples(): 
        df = frame_groups.get_group(int(row.frame)).reset_index(drop=True)
        idx_player_map = dict(zip(df.index, df.player_id))
        player_idx_map = dict(zip(df.player_id, df.index))
        idx_possessor = player_idx_map[row.player_id]
        positions = df[['x','y','vx','vy']].to_numpy()

        p0,v0 = positions[idx_possessor,:2],positions[idx_possessor,2:]
        in_cone = in_cone_vision(p0, v0, positions[:,:2], params['pitch_control']['vision_angle'])
        in_cone[idx_possessor] = False
        in_cone_index = [i for i,val in enumerate(in_cone) if val==True]
        in_cone_ids = [idx_player_map[i] for i,val in enumerate(in_cone) if val==True]

        if len(in_cone_ids) > 0:
            df_poss.at[row.Index, 'within_vision'] = in_cone_ids

        try:
            next_player = df_poss.loc[row.next_gain_event_id]
            recipients_ids = [next_player.player_id.split(';')[i] for i,team in enumerate(next_player.team_id.split(';')) if row.team_id == team]
            recipients_index = [player_idx_map[pid] for pid in recipients_ids]
            in_cone_ids = list(set(in_cone_ids) + set(recipients_ids))
            in_cone_index = list(set(in_cone_index) + set(recipients_index))
        except:
            pass


        if len(in_cone_ids) == 0:
            continue
        teammates = positions[in_cone_index,:2]

        ball_start_pos = (row.ball_x,row.ball_y)
        team_possession = team_map[row.team_possession]
        pc_results = pc.generate_pitch_control_for_passing_line(players_pc, row.frame, team_possession, 
                                                                 np.sign(params['active_goal'][row.team_possession][0]),
                                                                 ball_start_pos, teammates, pc_params, 
                                                                 granularity_m=params['pitch_control']['pc_granularity'])

        pc_gain = {in_cone_ids[i]:round(result[-1]-result[0],3) for i,(result,grid) in enumerate(pc_results)}
        safety = {in_cone_ids[i]:{'avg':round(np.mean(result),3),
                                  'min':round(np.min(result),3),
                                  'max':round(np.max(result),3)} for i,(result,grid) in enumerate(pc_results)}
        df_poss.at[row.Index, 'pc_gain'] = pc_gain
        df_poss.at[row.Index, 'pc_passing_line'] = safety
        

        return model.merge(df_poss[['frame','pc_gain','pc_passing_line','within_vision']], on='frame', how='left')


def postprocess(model, params, ball_data):
    """
    This routine postprocesses events data to add the following attributes. The details and conventions used
    may be found in the auto-eventing paper doi.org/10.1007/s12283-022-00381-6

    To possession gains:
    :distance_traveled_recipient: distance traveled by recipient from the time the preceding possession loss until possession gain
    :incoming_angle: Angle between incoming pass and current location
    :outgoing_angle: Angle between current location and next possession gain if possession ends in possession loss
    :player_location: Player location within the convex hull of other team
    :ball_from: Pass comes from teammate or opponent
    :packing_possession: How many opponents are overtaken by the possession starting in this possession gain
    :home_span_x/y, away_span_x/y: Span of home/away team in both x/y directions at the time of possession gain
    
    To passes:
    :outgoing_angle: Angle between pass location and next gain
    :player_location: Player location within the convex hull of other team
    :packing_pass: How many opponents are overtaken by the pass
    :home_span_x/y, away_span_x/y: Span of home/away team in both x/y directions at the time of reception
    :distance_traveled_ball_x/y: distance traveled by ball from the time of the pass is made until the subsequent possession gain

    :param model: Dataframe of events
    :param params: Dictionary containing all the relevant parameters
    :param ball_data: Dataframe of ball data
    :return: Dataframe of events with columns added
    """ 
    
    td = params['td'][['frame','player_id','team_id','x','y']]
    
    receptions = model[pd.eval("model.is_gain == True")].reset_index().rename(columns={'index':'reception_index'})
    
    model['distance_traveled_recipient'] = np.nan
    model['incoming_angle'] = np.nan
    model['outgoing_angle'] = np.nan
    model['player_location'] = ''
    model['ball_from'] = ''
    model['packing_pass'] = np.nan
    model['packing_possession'] = np.nan
    model['distance_traveled_ball_x'] = np.nan
    model['distance_traveled_ball_y'] = np.nan
    model['home_span_x'] = np.nan
    model['home_span_y'] = np.nan
    model['away_span_x'] = np.nan
    model['away_span_y'] = np.nan
    model.set_index('frame', inplace=True)
    goalies = params['goalkeeper_ids']
    player_ids = td['player_id'].unique().tolist()
    home_id = params['home_id']
    away_id = params['away_id']
    half = model.iloc[0]['half']
    segment_id = params['segment_id']
    goalies_by_team = {home_id: params['home_goalkeeper_ids'],
                       away_id: params['away_goalkeeper_ids'],}

    ############################################ ADD ATTRIBUTES TO POSSESSIONS ############################################

    for row in receptions.itertuples():
        if row.previous_loss_frame == '':
            continue
        prev_frame = int(row.previous_loss_frame)
        this_frame = row.frame
        this_player = row.player_id.split(';')[0]
        this_team = row.team_id.split(';')[0]
        active_goal = np.sign(params['active_goal'][this_team][0])
        other_team = home_id if this_team == away_id else away_id
        this_interval = td[pd.eval("td.frame >= prev_frame and td.frame <= this_frame")]
        this_interval_player = this_interval[pd.eval("this_interval.player_id == this_player")]
        
        ## Team of preceding pass
        prev_team = model.at[prev_frame,'team_id'].split(';')

        if this_team == prev_team[0]:
            model.at[this_frame,'ball_from'] = 'teammate'
        else:
            model.at[this_frame,'ball_from'] = 'opponent'
                

            
        ## Distance traveled by recipient since prior loss
        model.at[this_frame, 'distance_traveled_recipient'] = round(np.sqrt(this_interval_player.x.diff()**2 + this_interval_player.y.diff()**2).sum(),2)
        
        ## Convex hull and spatial span of both teams at reception time
        try:
            other_team_gain = this_interval[pd.eval("this_interval.frame == this_frame and this_interval.team_id == other_team")]
            index_other_infielder = [i for i, player in enumerate(other_team_gain['player_id'].tolist()) if player not in goalies_by_team[other_team]]
            hull_other = other_team_gain.iloc[index_other_infielder][['x','y']].to_numpy()
            span_other = (round(abs(hull_other[:,0].max()-hull_other[:,0].min()),2), round(abs(hull_other[:,1].max()-hull_other[:,1].min()),2))

            this_team_gain = this_interval[pd.eval("this_interval.frame == this_frame and this_interval.team_id == this_team")]
            index_this_infielder = [i for i, player in enumerate(this_team_gain['player_id'].tolist()) if player not in goalies_by_team[this_team]]
            hull_this = this_team_gain.iloc[index_this_infielder][['x','y']].to_numpy()
            span_this = (round(abs(hull_this[:,0].max()-hull_this[:,0].min()),2), round(abs(hull_this[:,1].max()-hull_this[:,1].min()),2))

            model.at[this_frame,'player_location'] = in_hull(this_interval_player.iloc[-1][['x','y']], other_team_gain, index_other_infielder, active_goal)
            if home_id == this_team:
                model.at[this_frame,'home_span_x'] = span_this[0]
                model.at[this_frame,'home_span_y'] = span_this[1]
                model.at[this_frame,'away_span_x'] = span_other[0]
                model.at[this_frame,'away_span_y'] = span_other[1]
            else:
                model.at[this_frame,'home_span_x'] = span_other[0]
                model.at[this_frame,'home_span_y'] = span_other[1]
                model.at[this_frame,'away_span_x'] = span_this[0]
                model.at[this_frame,'away_span_y'] = span_this[1]
        except:
            model.at[this_frame,'player_location'] = ''
            model.at[this_frame,'home_span_x'] = np.nan
            model.at[this_frame,'home_span_y'] = np.nan
            model.at[this_frame,'away_span_x'] = np.nan
            model.at[this_frame,'away_span_y'] = np.nan
            
        ## Packing metric: How many opponents are overtaken by this possession
        if row.next_loss_frame == '':
            continue
        next_frame = int(row.next_loss_frame)
        td_loss_frame = td[pd.eval("td.frame == next_frame")]
        other_team_loss = td_loss_frame[pd.eval("td_loss_frame.team_id == other_team")]
        try:
            x_loss = td_loss_frame[pd.eval("td_loss_frame.player_id == this_player")].iloc[0]['x']
            x_gain = this_interval[pd.eval("this_interval.player_id == this_player and this_interval.frame == this_frame")].iloc[0]['x']
            
            if active_goal > 0:
                opponents_behind_0 = (other_team_gain['x'] < x_gain).sum()
                opponents_behind_1 = (other_team_loss['x'] < x_loss).sum()
            else:
                opponents_behind_0 = (other_team_gain['x'] > x_gain).sum()
                opponents_behind_1 = (other_team_loss['x'] > x_loss).sum()
            model.at[this_frame,'packing_possession'] = opponents_behind_1 - opponents_behind_0
            
        except:
            model.at[this_frame,'packing_possession'] = np.nan
        
        ## Angle
        try:
            prev_ball = model.at[prev_frame,'ball_pos']
        except:
            prev_dir = model.at[this_frame,'ball_dir_0']
            prev_ball = (this_ball[0]-prev_dir[0], this_ball[1]-prev_dir[1])  
        this_ball = model.at[this_frame,'ball_pos']
        model.at[this_frame,'incoming_angle'] = compute_angle(prev_ball, this_ball, active_goal)
        if row.possession_outcome == 'pass':
            this_ball = model.at[next_frame,'ball_pos']
            next_gain = model.at[next_frame,'next_gain_frame']
            try:
                next_ball = model.at[next_gain,'ball_pos']
            except:
                next_dir = model.at[next_frame,'ball_dir_1']
                next_ball = (this_ball[0]+next_dir[0], this_ball[1]+next_dir[1])     
            model.at[this_frame,'outgoing_angle'] = compute_angle(next_ball, this_ball, active_goal)
            
    ############################################ ADD ATTRIBUTES TO PASSES ############################################
    passes = model[pd.eval("model.event_name.str.contains('|'.join(['pass','cross','shot','deflect']))")].reset_index().rename(columns={'index':'pass_index'})
    for row in passes.itertuples():
        
        this_frame = row.frame
        this_player = row.player_id.split(';')[0]
        this_team = row.team_id.split(';')[0]
        other_team = home_id if this_team == away_id else away_id
        active_goal = np.sign(params['active_goal'][this_team][0])

        this_frame_data = td[pd.eval("td.frame == this_frame")]
        try:
            this_pos = this_frame_data[pd.eval("this_frame_data.player_id == this_player")].iloc[0][['x','y']].to_numpy()
        except:
            this_pos = np.asarray(eval(row.xy.split(';')[0]))
            
        
        ## Convex hull and spatial span of both teams at pass time
        try:
            other_team_this_frame = this_frame_data[pd.eval("this_frame_data.team_id == other_team")]
            index_other_infielder = [i for i, player in enumerate(other_team_this_frame['player_id'].tolist()) if player not in goalies_by_team[other_team]]
            hull_other = other_team_this_frame.iloc[index_other_infielder][['x','y']].to_numpy()
            span_other = (round(abs(hull_other[:,0].max()-hull_other[:,0].min()),2), round(abs(hull_other[:,1].max()-hull_other[:,1].min()),2))

            this_team_this_frame = this_frame_data[pd.eval("this_frame_data.team_id == this_team")]
            index_this_infielder = [i for i, player in enumerate(this_team_this_frame['player_id'].tolist()) if player not in goalies_by_team[this_team]]
            hull_this = this_team_this_frame.iloc[index_this_infielder][['x','y']].to_numpy()
            span_this = (round(abs(hull_this[:,0].max()-hull_this[:,0].min()),2), round(abs(hull_this[:,1].max()-hull_this[:,1].min()),2))

            model.at[this_frame,'player_location'] = in_hull(this_pos, other_team_this_frame, index_other_infielder, active_goal)
            if home_id == this_team:
                model.at[this_frame,'home_span_x'] = span_this[0]
                model.at[this_frame,'home_span_y'] = span_this[1]
                model.at[this_frame,'away_span_x'] = span_other[0]
                model.at[this_frame,'away_span_y'] = span_other[1]
            else:
                model.at[this_frame,'home_span_x'] = span_other[0]
                model.at[this_frame,'home_span_y'] = span_other[1]
                model.at[this_frame,'away_span_x'] = span_this[0]
                model.at[this_frame,'away_span_y'] = span_this[1]
        except:
            model.at[this_frame,'player_location'] = ''
            model.at[this_frame,'home_span_x'] = np.nan
            model.at[this_frame,'home_span_y'] = np.nan
            model.at[this_frame,'away_span_x'] = np.nan
            model.at[this_frame,'away_span_y'] = np.nan
            
        ## Distance traveled by ball and angle
        if np.isnan(row.outgoing_angle):
            this_ball = model.at[this_frame,'ball_pos']
            try:
                next_ball = model.at[row.next_gain_frame,'ball_pos']
            except:
                next_dir = model.at[this_frame,'ball_dir_1']
                next_ball = (this_ball[0]+next_dir[0], this_ball[1]+next_dir[1])     
            model.at[this_frame,'outgoing_angle'] = compute_angle(next_ball, this_ball, active_goal)
            
        if row.next_gain_frame == '':  
            if row.event_outcome == 'dead ball':
                if ball_data.at[this_frame,'ball_status'] == 'dead':
                    ### Passes that have been imputed
                    next_frame = this_frame
                    next_pos = this_pos
                else:                    
                    next_frame = (ball_data.loc[this_frame:,'ball_status'] == 'dead').idxmax()-1
                    next_pos = (ball_data.at[next_frame,'ball_x'], ball_data.at[next_frame,'ball_y'])
            else:
                continue
        else:
            next_frame = int(row.next_gain_frame)
            next_pos = model.at[next_frame,'ball_pos']
        this_pos = model.at[this_frame,'ball_pos']
        model.at[this_frame,'distance_traveled_ball_x'] =  round(next_pos[0]-this_pos[0],2)
        model.at[this_frame,'distance_traveled_ball_y'] =  round(next_pos[1]-this_pos[1],2)
        
        if row.event_outcome in ['goal','end of half']:
            continue
            
        next_frame_data = td[pd.eval("td.frame == next_frame")]
        
        ## Packing metric: How many opponents are overtaken by this pass
        try:
            x_loss = this_pos[0]
            x_gain = next_pos[0]
            other_team_loss = this_frame_data[pd.eval("this_frame_data.team_id != this_team")]
            other_team_gain = next_frame_data[pd.eval("next_frame_data.team_id != this_team")]

            if active_goal > 0:
                opponents_behind_0 = (other_team_loss['x'] < x_loss).sum()
                opponents_behind_1 = (other_team_gain['x'] < x_gain).sum()
            else:
                opponents_behind_0 = (other_team_loss['x'] > x_loss).sum()
                opponents_behind_1 = (other_team_gain['x'] > x_gain).sum()
            model.at[this_frame,'packing_pass'] = opponents_behind_1 - opponents_behind_0
            
        except:
            model.at[this_frame,'packing_pass'] = np.nan
        
    
    ## Add convex hull and spatial span of teams for frames with losses that have not been yet accounted for
    next_loss_frames = model.query("next_loss_frame != ''")['next_loss_frame'].astype('int64').tolist()
    previous_loss_frames = model.query("previous_loss_frame != ''")['previous_loss_frame'].astype('int64').tolist()
    loss_frames = list(set(next_loss_frames + previous_loss_frames))
    loss_frames.sort()
    
    for frame in loss_frames:
        
        if not np.isnan(model.at[frame, 'home_span_x']):
            continue

        try:
            
            home_this = td[pd.eval("td.team_id == home_id and td.frame == frame")]
            index_home_infielder = [i for i, player in enumerate(home_this['player_id'].tolist()) if player not in goalies_by_team[home_id]]
            hull_home = home_this.iloc[index_home_infielder][['x','y']].to_numpy()

            away_this = td[pd.eval("td.team_id == away_id and td.frame == frame")]
            index_away_infielder = [i for i, player in enumerate(away_this['player_id'].tolist()) if player not in goalies_by_team[away_id]]
            hull_away = away_this.iloc[index_away_infielder][['x','y']].to_numpy()

            model.at[frame,'home_span_x'] = round(abs(hull_home[:,0].max()-hull_home[:,0].min()),2)
            model.at[frame,'home_span_y'] = round(abs(hull_home[:,1].max()-hull_home[:,1].min()),2)
            model.at[frame,'away_span_x'] = round(abs(hull_away[:,0].max()-hull_away[:,0].min()),2)
            model.at[frame,'away_span_y'] = round(abs(hull_away[:,1].max()-hull_away[:,1].min()),2)
        except:
            model.at[frame,'home_span_x'] = np.nan
            model.at[frame,'home_span_y'] = np.nan
            model.at[frame,'away_span_x'] = np.nan
            model.at[frame,'away_span_y'] = np.nan

    model.reset_index(inplace=True)  
    model.drop(columns=['next_loss_frame',
                          'previous_loss_frame',
                          'next_gain_frame'], inplace=True)
    return model



def set_event_logic(model, params):

    """
    This routine corrects sequences where a player from team A last touched the ball before OOB and the play was resumed by team A. It finds
    the last instance where a player from team B was in possession of the ball (either contested or not) and sets all actions from that point
    until OOB as `no control`

    :param model: Events dataframe
    :param params: Dictionary containing all the relevant parameters        
    :return: Events dataframe
    """ 

    fsp = model[pd.eval("model.from_set_piece in ['throw-in','goal kick','corner kick']")].reset_index()
    oob = model[pd.eval("model.dead_ball_event in ['out for throw-in','out for goal kick','out for corner kick']")].reset_index()
    model.set_index('frame', inplace=True)
    event_logic_frames = params['event_logic_lookbehind']*params['fps']
    for row in fsp.itertuples():
        team_set_piece = row.team_id.split(';')[0]
        other_team = params['home_id'] if team_set_piece == params['away_id'] else params['away_id']
        f_last_oob = oob.at[row.Index,'frame']


        if  ((model.loc[f_last_oob-event_logic_frames:f_last_oob,'team_id'].str.contains(other_team)) & 
             (model.loc[f_last_oob-event_logic_frames:f_last_oob,'ball_control'] != 'no control')).any():

            f_last_touch = ((model.loc[f_last_oob-event_logic_frames:f_last_oob,'team_id'].str.contains(other_team)) & 
                            (model.loc[f_last_oob-event_logic_frames:f_last_oob,'ball_control'] != 'no control'))[::-1].idxmax()

            event_found = False
            this_event = ''
            for f in model.loc[f_last_touch+1:f_last_oob].itertuples():
                if f.ball_control != 'no control' and team_set_piece in f.team_id:
                    if f.event_name != '':
                        event_found = True
                        this_event = f.event_name
                    model.at[f.Index,'step_change'] += '-18'
                    model.at[f.Index,'ball_control'] = 'no control'
                    model.at[f.Index,'event_name'] = ''

            if model.at[f_last_touch,'event_name'] == '' and event_found:
                model.at[f_last_touch,'event_name'] = this_event
                model.at[f_last_touch,'step_change'] += '-18'
                if model.at[f_last_touch,'team_id'].split(';')[0] == team_set_piece:
                    index_first = model.at[f_last_touch,'team_id'].split(';').index(other_team)
                    for col in ['player_id','team_id','xy','distance']:
                        tm = model.at[f_last_touch, col].split(';')
                        model.at[f_last_touch, col] = ';'.join([tm[index_first]] + tm[:index_first] + tm[index_first+1:])


    return model.reset_index()



def set_event_ballcontrol(row):
    """
    This routine establishes the ball_control/event_name columns based on the event_name

    :param row: Row in the events dataframe
    :return: Row updated with ball_control value
    """ 

    if row['event_name'] == 'possession':
        row['ball_control'] = 'possession'
        row['event_name'] = ''

    elif row['event_name'] == 'no control':
        row['ball_control'] = 'no control'
        row['event_name'] = ''

    elif any(ele in row['event_name'] for ele in ['pass','shot','reception','save','claim','own','cross','interception']):
        row['ball_control'] = 'possession'

    if row['contest'] == '':
        row['contest_player_id'] = ''
        row['contest_team_id'] = ''
        row['contest_distance'] = ''
        row['contest_xy'] = ''


    return row



def raw_defensive_mode(defender_frames, set_piece, def_team, def_params):
    """
    This routine applies the defensive mode raw definitions based on the locations of the players 

    :param defender_frames: Locations of defenders
    :param set_piece: String to identify if this is a set piece
    :param def_team: String of defending team ID
    :param def_params: Dictionary containing all the relevant defensive modes parameters    
    :return: Defensive mode string without context
    """ 


    pitch_length = def_params['pitch_length']
    active_goal = def_params['active_goal']

    #Check ball status
    if defender_frames.iloc[0]['ball_status'] == 'dead':
        return ''
    
    #Determine which goal is being defended and ball position using active_goal parameter
    ball_x = defender_frames.iloc[0]['ball_x']
    player_x = defender_frames.x.to_numpy()
    if np.sign(active_goal[def_team][0]) < 0: #flip so attack is from left to right for convenience
        ball_x *= -1
        player_x *= -1

    num_defenders_behind_ball = np.sum(np.where(player_x < ball_x,1,0))
    
    #Set piece defense: When a set piece in the own half occurs (corner, throw-in, free kick), assign Set
    #piece defense until the ball goes out-of-play or possession is regained
    if set_piece in ['free kick', 'corner kick','penalty kick'] and ball_x <- pitch_length/6:
        return 'set-piece-defense'
    
    
    if is_high_press(player_x, def_params):
        if set_piece == 'goal kick': #check for goal kick
            if is_gk_press(player_x, def_params):
                return 'goal-kick-press'
        return 'high-press'
    
    elif is_mid_block(player_x, def_params):
        return 'mid-block'

    elif is_low_block(player_x, def_params):
        if is_deep_low_block(player_x, def_params):
            return 'low-block-deep'

        elif is_advanced_low_block(player_x, def_params):
            return 'low-block-advanced'

        return 'low-block'
    return 'UNCLEAR'

        


def set_defensive_modes(model, team_possession, params):
    """
    This routine generates the defensive modes based on the player locations, the team possession and events. 
    Specific parts of the routine are commented for better explainability. 

    :param model: Events dataframe
    :param team_possession: Team possession dataframe
    :param params: Dictionary containing all the relevant parameters    
    :return: Defensive modes dataframe
    """ 

    model.set_index('frame', inplace=True)

    home_id = params['home_id']
    away_id = params['away_id']
    pitch_length = params['pitch_length']
    active_goal = params['active_goal']    
    td = params['td']
    def_params = params['defensive_modes']
    def_params['pitch_length'] = pitch_length
    def_params['active_goal'] = active_goal
    def_params['pitch_measures'] = params['pitch_measures']

    frame_poss_map = dict(zip(team_possession['frame'],team_possession['defending_team_id']))
    td['defending_team_id'] = td['frame'].map(frame_poss_map)
    td['is_defending'] = np.where(td.defending_team_id==td.team_id,1,0)
    frames = team_possession['frame'].tolist()

    td_def_gb = dict(tuple(td[pd.eval("td.is_defending == 1 and td.player_id not in params['goalkeeper_ids']")][['frame','x','ball_x','ball_status']].groupby('frame')))
    context_modes = {}
    raw_modes = {}
    prev_mode = None
    prev_team = team_possession.iloc[0]['defending_team_id']
    transition_period = False
    transition_t0 = None
        
    
    team_possession['is_home'] = team_possession['defending_team_id'].map({home_id:1,
                                                                           away_id:0})
    team_possession['change'] = team_possession['is_home'].diff()
    team_possession.at[0,'change'] = 0    
    change_frame = team_possession.query("change != 0")['frame'].tolist()
    change_frame.append(np.inf)
    change_frame.sort()


    team_poss_data = team_possession[['frame','from_set_piece','defending_team_id','time_ms']].set_index('frame').to_dict(orient='index')
    pressure = model['pressure'].to_dict()
    ## Raw extraction of defensive modes based on definitions
    for frame in frames:
        defender_frames = td_def_gb[frame]
        set_piece, team, time_ms = team_poss_data[frame]['from_set_piece'], team_poss_data[frame]['defending_team_id'], team_poss_data[frame]['time_ms']

        preliminary_defensive_mode = raw_defensive_mode(defender_frames, set_piece, team, def_params) 
        raw_modes[frame] = preliminary_defensive_mode
        defensive_mode = ''
        
        if preliminary_defensive_mode == '':
            ## Handle dead ball
            pass
        else:
            ## In here beyond the definitions we add context, for instance pressure or defensive transition
            ## Check if we need to break out of a transition period (it has been transition_period_duration seconds since last change of possession)
            if transition_period and time_ms-transition_t0>def_params['transition_period_duration']*1000:
                transition_period = False
                transition_t0 = None

            ## If the last frame was set-piece defense and possession has not changed and we have not encountered a dead ball, then we continue set-piece defense
            if prev_mode == 'set-piece-defense' and team == prev_team and preliminary_defensive_mode != '': 
                defensive_mode = 'set-piece-defense'

            ## If there was a change of possession from the last frame
            elif team != prev_team:
                ## Look for the next change_frame as the end of possession, and subtract one to find the last possession frame of the team
                transition_period = True 
                transition_t0 = time_ms 
                end_of_possession = change_frame[[True if frame >= f and frame < change_frame[i+1] else False for i,f in enumerate(change_frame)].index(True)+1]-1
                end_of_transition = int(frame+def_params['transition_period_duration']*params['fps'])
                end_pressure = min(end_of_transition, end_of_possession)
                pressure_interval = [pressure.get(f,'') for f in range(frame, end_pressure+1)]
                pressure_interval = [item for sublist in pressure_interval for item in sublist]
                players_pressing = set(filter(lambda x: x != '', pressure_interval))
                if len(players_pressing) > 1: ## check for press after loss for at least two players, otherwise defensive-transition
                    defensive_mode = 'press-after-loss'
                else:                
                    defensive_mode = 'defensive-transition'

            ## If there was not a change of possession from the last frame, but we are in a transition period
            elif transition_period: ## if less than three seconds since change of poss, we still mark transition
                if len(players_pressing) > 1: ## check for press after loss, otherwise defensive-transition
                    defensive_mode = 'press-after-loss'
                else:                
                    defensive_mode = 'defensive-transition'

            ## If we are not in a transition period, we just use the preliminary mode
            else:
                defensive_mode = preliminary_defensive_mode
                ## If high press but off a goal kick, mark goal-kick-press    
                if defensive_mode == 'high-press' and prev_mode == 'goal-kick-press':
                    defensive_mode = 'goal-kick-press'

        prev_mode = defensive_mode 
        prev_team = team
        context_modes[frame] = defensive_mode

    team_possession.reset_index(inplace=True)
    def_modes = team_possession[['frame','half','time_ms','minute','sec','ball_status',
                                'from_set_piece','defending_team_id','attacking_team_id']].copy()
    def_modes['defensive_mode_raw'] = def_modes['frame'].map(raw_modes)
    def_modes['defensive_mode_context'] = def_modes['frame'].map(context_modes)


    ## Routine to fill `unclears` based on the average player location values of the frames where definitions are satisfied
    home = td[pd.eval("td.is_home == 1 and td.ball_status == 'in_play' and td.player_id not in params['goalkeeper_ids']")].copy()
    away = td[pd.eval("td.is_home == 0 and td.ball_status == 'in_play' and td.player_id not in params['goalkeeper_ids']")].copy()
    if active_goal[home_id][0] > 0:
        away['x'] = -away['x']
        away['y'] = -away['y']

    elif active_goal[away_id][0] < 0:
        home['x'] = -home['x']
        home['y'] = -home['y']

    modes_to_fill = ['low-block-deep','low-block','low-block-advanced','mid-block','high-press']
    modes_to_learn = modes_to_fill + ['UNCLEAR']
    train = def_modes[pd.eval("def_modes.defensive_mode_context in modes_to_learn")].copy()
    home_defending = train[pd.eval("train.attacking_team_id == away_id")]['frame'].tolist()
    away_defending = train[pd.eval("train.attacking_team_id == home_id")]['frame'].tolist()

    home_def = home[pd.eval("home.frame in home_defending")][['frame','x']].groupby('frame').agg('median').reset_index()
    home_def['defending_team_id'] = home_id
    away_def = away[pd.eval("away.frame in away_defending")][['frame','x']].groupby('frame').agg('median').reset_index()
    away_def['defending_team_id'] = away_id

    both_def = pd.concat([home_def, away_def], ignore_index=True).sort_values(by='frame')
    train = train[['frame','defensive_mode_context']].merge(both_def, on='frame').rename(columns={'x':'med_x'})
    unclears = train[pd.eval("train.defensive_mode_context == 'UNCLEAR'")]
    def_modes['defensive_mode_filled'] = def_modes['defensive_mode_context'].copy()

    bins = round(pitch_length-1)
    home_hist = {}
    away_hist = {}
    for mode in modes_to_fill:
        
        home_hist[mode], bin_edges = np.histogram(train.query("defending_team_id == @home_id and defensive_mode_context == @mode")['med_x'],
                bins=bins, range=(-pitch_length/2, pitch_length/2), density=True)
        away_hist[mode], bin_edges = np.histogram(train.query("defending_team_id == @away_id and defensive_mode_context == @mode")['med_x'],
                bins=bins, range=(-pitch_length/2, pitch_length/2), density=True)
                                       
    def_modes.set_index('frame', inplace=True)
    for row in unclears.itertuples():
        loc = [True if row.med_x >= left and row.med_x < bin_edges[i+1] else False for i,left in enumerate(bin_edges[:-1],0)].index(True)
        if row.defending_team_id == home_id:
            likeliest_mode = modes_to_fill[np.argmax([home_hist[mode][loc] for mode in modes_to_fill])]
        elif row.defending_team_id == away_id:
            likeliest_mode = modes_to_fill[np.argmax([away_hist[mode][loc] for mode in modes_to_fill])]
        
        def_modes.at[row.frame,'defensive_mode_filled'] = likeliest_mode

    def_modes.reset_index(inplace=True)  
    model.reset_index(inplace=True)

    ## Collapse intervals shorter than a threshold
    def_modes['defensive_mode'] = collapse_short_intervals(def_modes, def_params, params)

    ## Depopulate dead ball frames
    def_modes.loc[def_modes.defensive_mode_raw=='','defensive_mode'] = ''

    return def_modes



def collapse_short_intervals(df, def_params, params):
    """
    This routine returns series for collapsed modes at each frame. Works by looking for intervals less than the given thresh_frame_count and 
    searched outward for the nearest long interval, whose label is taken and given to the newly-collapsed interval

    :param df: Defensive modes dataframe
    :param def_params: Dictionary containing all the relevant defensive modes parameters    
    :param params: Dictionary containing all the relevant parameters    
    :return: Defensive modes dataframe
    """ 

    thresh_frame_count = def_params['min_interval_duration']*params['fps']
    prev_team = df.defending_team_id.iloc[0]
    prev_mode = df.defensive_mode_filled.iloc[0]
    start,end = 0,None
    intervals = {}

    for row in df.itertuples():
        frame = row.frame
        team = row.defending_team_id
        mode = row.defensive_mode_filled

        #If the previous interval ended (you encountered a new mode or team)
        #add to the dictionary
        if mode != prev_mode or team!=prev_team :
            end = frame - 1
            intervals[(start,end)] = (prev_team,prev_mode)
            start = frame #reset the start idx

        prev_team = team
        prev_mode = mode

    intervals_list = [i+v for i,v in intervals.items()]
    revisions = []
    forbidden_modes = ['UNCLEAR', 'defensive-transition','press-after-loss','set-piece-defense','goal-kick-press','']
    for i,v in enumerate(intervals_list):
        start,end,defending_team,mode = v
        if end-start <= thresh_frame_count and mode not in forbidden_modes: #if the interval is leq 3s and the mode is unclear
    
            #search outward, making sure we are staying within the range where the team is the same

            for j in range(1,len(intervals_list)):

                if i-j in intervals_list:
                    before_interval = intervals_list[i-j]
                    before_team_same =  before_interval[2] == defending_team
                else:
                    before_interval = None

                if i+j in intervals_list:
                    after_interval = intervals_list[i+j]
                    after_team_same = after_interval[2] == defending_team
                else:
                    after_interval = None
                
                
                if before_interval is not None and before_interval[1]-before_interval[0]>thresh_frame_count and before_team_same:
                    before_mode = before_interval[-1]
                    if before_mode not in forbidden_modes:
                        revisions.append((start,end,defending_team,before_mode))
                        break

                if after_interval is not None and after_interval[1]-after_interval[0]>thresh_frame_count and after_team_same:
                    after_mode = after_interval[-1]
                    if after_mode not in forbidden_modes:
                        revisions.append((start,end,defending_team,after_mode))
                        break

    revisions = {(v[0],v[1]):(v[2],v[3]) for v in revisions}
    intervals.update(revisions)
    new_map = {}
    for i in intervals:
        for k in range(i[0],i[1]+1):
            new_map[k] = intervals[i][-1]
    
    return df.frame.map(new_map)


def is_high_press(player_xs, def_params):
    """
    This routine returns boolean indicating if defense is in high press.

    :param player_xs: x-coordinates of defenders
    :param def_params: Dictionary containing all the relevant defensive modes parameters    
    :return: True if high press
    """ 
    pitch_measures = def_params['pitch_measures']
    num_defenders_more_advanced = np.sum(np.where(player_xs >= pitch_measures['arc_dim']/2,1,0))

    if num_defenders_more_advanced < def_params['high_press_min_advanced_defenders']:
        return False
    num_defenders_inzone = np.sum(np.where((player_xs <= def_params['pitch_length']/2 - pitch_measures['smbx_dim']) & (player_xs >= -pitch_measures['arc_dim']),1,0))
    return num_defenders_inzone == len(player_xs)
   
def is_gk_press(player_xs, def_params):
    """
    This routine returns boolean indicating if defense is in goal-kick press.

    :param player_xs: x-coordinates of defenders
    :param def_params: Dictionary containing all the relevant defensive modes parameters   
    :return: True if goal-kick press
    """ 
    pitch_measures = def_params['pitch_measures']
    num_defenders_inzone = np.sum(np.where((player_xs <= def_params['pitch_length']/2 - pitch_measures['smbx_dim']) & (player_xs >= -pitch_measures['arc_dim']),1,0))
    return num_defenders_inzone == len(player_xs)

def is_mid_block(player_xs, def_params):
    """
    This routine returns boolean indicating if defense is in mid block.

    :param player_xs: x-coordinates of defenders
    :param def_params: Dictionary containing all the relevant defensive modes parameters  
    :return: True if mid block
    """ 
    num_defenders_opp_half = np.sum(np.where(player_xs > 0,1,0))
    if num_defenders_opp_half<def_params['mid_block_min_defenders_opp_half']:
        return False
    #All players in specified zone
    num_defenders_inzone = np.sum(np.where((player_xs >= -def_params['pitch_length']/2*def_params['mid_block_pitch_fraction']) & (player_xs<=def_params['pitch_length']/2*def_params['mid_block_pitch_fraction']),1,0))
    return num_defenders_inzone == len(player_xs)


def is_low_block(player_xs, def_params):
    """
    This routine returns boolean indicating if defense is in low block.

    :param player_xs: x-coordinates of defenders
    :param def_params: Dictionary containing all the relevant defensive modes parameters    
    :return: True if low block
    """ 
    num_defenders_opp_half = np.sum(np.where(player_xs > 0,1,0))
    if num_defenders_opp_half>def_params['low_block_max_defenders_opp_half']:
        return False
    #All players in specified zone
    num_defenders_inzone = np.sum(np.where(player_xs <= def_params['pitch_measures']['arc_dim'],1,0))
    return num_defenders_inzone == len(player_xs)

def is_deep_low_block(player_xs, def_params):
    """
    This routine returns boolean indicating if defense is in deep low block.

    :param player_xs: x-coordinates of defenders
    :param def_params: Dictionary containing all the relevant defensive modes parameters    
    :return: True if deep low block
    """ 
    num_defenders_inzone = np.sum(np.where(player_xs <= -def_params['pitch_measures']['arc_dim'],1,0))
    return num_defenders_inzone == len(player_xs)

def is_advanced_low_block(player_xs, def_params):
    """
    This routine returns boolean indicating if defense is in advanced low block.

    :param player_xs: x-coordinates of defenders
    :param def_params: Dictionary containing all the relevant defensive modes parameters    
    :return: True if advanced low block
    """ 
    
    pitch_measures = def_params['pitch_measures']
    num_defenders_inzone = np.sum(np.where((player_xs <= pitch_measures['arc_dim']) & (player_xs >= (-def_params['pitch_length']/2+pitch_measures['penbox_len'])/2),1,0))
    return num_defenders_inzone == len(player_xs)



def set_pressure(model, team_possession, params):

    """
    This routine generates pressures. The pressures are generated for each particular frame, and then they are transformed to team pressures
    using the create_team_pressures routine.  

    A NEW VERSION OF THIS ROUTINE IS CURRENTLY BEING TESTED IN new_developments/pressures_test.py. Raúl Peláez was the responsible of validating the pressures,
    he should be the one to greenlight the transition to the new function.    

    :param model: Events dataframe
    :param team_possession: Team possession dataframe
    :param params: Dictionary containing all the relevant parameters    
    :return: Events and team possession dataframes updated, team pressures dataframe
    """     

    possession = model.query("ball_control == 'possession' and contest == '' and from_set_piece == '' and team_id == team_possession").copy()
    contest = model.query("contest in ['ground duel','contest'] and from_set_piece == ''").copy()
    possession = pd.concat([possession, contest], ignore_index=True).sort_values(by='frame',ascending=True)
    possession.set_index('frame', drop=False,inplace=True)

    possession['pressure'] = ''
    possession['pressure_distance'] = ''

    pressure_params = params['pressure']
    td = params['td']
    
    ## Create tracking data group by frame for faster access
    first_frame = td.iloc[0]['frame']
    new_frames = list(possession['frame'])
    old_frames = [max(f-int(pressure_params['moving_closer']*params['fps']), first_frame) for f in new_frames]
    all_frames = new_frames + old_frames
    home_grouped_now = dict(tuple(td[pd.eval("td.frame in new_frames and td.is_home == 1 and ((td.distance < pressure_params['radius'] and td.speed>pressure_params['min_speed']) or td.distance < pressure_params['radius_back'])")][['frame','player_id','x','y']].groupby('frame')))
    away_grouped_now = dict(tuple(td[pd.eval("td.frame in new_frames and td.is_home == 0 and ((td.distance < pressure_params['radius'] and td.speed>pressure_params['min_speed']) or td.distance < pressure_params['radius_back'])")][['frame','player_id','x','y']].groupby('frame')))
    all_grouped_past = dict(tuple(td[pd.eval("td.frame in old_frames")][['frame','x','y','player_id']].groupby('frame')))
    all_grouped_now = dict(tuple(td[pd.eval("td.frame in new_frames")][['frame','speed','player_id','vx','vy']].groupby('frame')))
    
    ball_z_all = td.drop_duplicates(subset=['frame']).set_index('frame')['ball_z']

    ## Loop over every frame where a player was in possession
    for current_event in possession.itertuples():

        active_goal = np.sign(params['active_goal'][current_event.team_id.split(';')[0]][0])
        current_player_id = current_event.player_id.split(';')[0]
        current_team = current_event.team_id.split(';')[0]
        current_is_home = params['players_df'].at[current_player_id,'is_home']
        attacker_xy = np.asarray((float(current_event.x.split(';')[0]), float(current_event.y.split(';')[0])))
        current_time = current_event.time_ms
        
        discrete = []
        continuous = []
        distances = []
        is_press = False
        height_condition = False
        ball_z_next_sec = None

        ## Special treatment for goalkeeper being pressed
        if current_player_id in params['goalkeeper_ids']: 

            future_poss = possession.loc[current_event.frame:current_event.frame+pressure_params['gk_possession_lookahead']*params['fps']].iloc[-1]
            if future_poss.player_id.split(';')[0] == current_player_id:
                ball_z_next_sec = ball_z_all.loc[current_event.frame:current_event.frame+pressure_params['gk_possession_lookahead']*params['fps']].mean()
            else:
                ball_z_next_sec = current_event.ball_z
            height_condition = ball_z_next_sec > pressure_params['gk_z_limit'][0] and ball_z_next_sec <= pressure_params['gk_z_limit'][1]
        
        goalie_hands = current_player_id in params['goalkeeper_ids'] and height_condition and in_penalty_box(current_event.ball_x, current_event.ball_y, -active_goal, params, tol=params['in_area'])

        ## Heuristic to discard instances where the goalkeeper has the ball in hands, because it is not a valid press
        if not goalie_hands:              
            if current_event.contest != '':       
                ## If the possession is contested, the presser is the player furthest from the ball                          
                p1,p2 = current_event.contest_player_id.split(';')
                t1,t2 = current_event.contest_team_id.split(';')
                
                if t1 == current_event.team_possession:
                    continuous.append(p2)
                else:
                    continuous.append(p1)
                
                p1x,p2x = current_event.contest_x.split(';')
                p1x = float(p1x)
                p2x = float(p2x)
                p1y,p2y = current_event.contest_y.split(';')
                p1y = float(p1y)
                p2y = float(p2y)
                distances.append(round(np.sqrt((p1x-p2x)**2+(p1y-p2y)**2),2))
                
            else:
                ## In every other case, we examine the opponents that have come closer to the ball handler in the last `moving_closer`
                last_poss_frame = max(current_event.frame-int(pressure_params['moving_closer']*params['fps']), first_frame)

                if current_is_home == 1 and current_event.frame in away_grouped_now:
                    pressers_current = away_grouped_now[current_event.frame].set_index('player_id').copy()
                    is_press = True
                elif current_is_home == 0 and current_event.frame in home_grouped_now:
                    pressers_current = home_grouped_now[current_event.frame].set_index('player_id').copy()
                    is_press = True

                if is_press: 
                    ## If there is at least one defender pressing
        
                    all_players_past = all_grouped_past[last_poss_frame].set_index('player_id').to_dict(orient='index')
                    all_players_now = all_grouped_now[current_event.frame].set_index('player_id').to_dict(orient='index')
                    try:
                        attacker_past_xy = (all_players_past[current_player_id]['x'],
                                            all_players_past[current_player_id]['y'])
                    except:
                        ## If the player in possession was not appearing before (may be subbed in, may have just reappeared), considered a pressure
                        attacker_past_xy = (np.inf, np.inf)
                        
                    pressers_xy = pressers_current[['x','y']].to_numpy()
                    pressers_current['distance_p2p'] = np.sqrt((pressers_xy[:,0] - attacker_xy[0])**2+(pressers_xy[:,1] - attacker_xy[1])**2)
                    pressers_current_distance = pressers_current['distance_p2p'].to_dict()
                    pressers_current_x = pressers_current['x'].to_dict()
                    pressers_current_y = pressers_current['y'].to_dict()
                    
                    if all_players_now[current_player_id]['speed'] > 0:
                        dir_attacker = np.asarray((all_players_now[current_player_id]['vx'],
                                                   all_players_now[current_player_id]['vy']))/all_players_now[current_player_id]['speed']                        
                    else:
                        dir_attacker = np.asarray((0,0))
                    ortho_attacker = np.asarray((-dir_attacker[1], dir_attacker[0]))
                    
                    ## Iterate through opponents
                    for presser_id,distance_p2p  in pressers_current_distance.items():
                        
                        presser_xy = np.asarray((pressers_current_x[presser_id], pressers_current_y[presser_id]))
                        
                        ## Check if presser is within sight of possessor
                        v_attacker_presser = presser_xy - attacker_xy
                        
                        if np.dot(v_attacker_presser, dir_attacker) >= 0:
                            ## if the presser is in front of the attacker, they will always be by construction within the radius
                            within_radius = True
                        else:
                            ## if the presser is in the back of the attacker, check if they are within the semiellipse of semiaxes
                            ## b=radius (in the direction orthogonal of the attacker's orientation) and
                            ## a=radius_back (direction of attacker's orientation)
                            proj_a = np.dot(v_attacker_presser, dir_attacker)
                            proj_b = np.dot(v_attacker_presser, ortho_attacker)
                            within_radius = (proj_a / pressure_params['radius_back']) ** 2 + (proj_b / pressure_params['radius']) ** 2 <= 1
    
                        v_presser_attacker = -v_attacker_presser/distance_p2p
                        dir_presser = np.asarray((all_players_now[presser_id]['vx'],
                                                  all_players_now[presser_id]['vy']))/all_players_now[presser_id]['speed']                        

                        ## Check if presser is moving in towards the attacker (+- a certain angle)
                        ## Instead of the attacker's current position, we pick the next position based on the attacker's current position and speed
                        if distance_p2p <= pressure_params['radius_back']:
                            towards_attacker = True
                        else:
                            attacker_next = attacker_xy + dir_attacker*all_players_now[current_player_id]['speed']/params['fps']
                            v_presser_attacker_next = attacker_next - presser_xy
                            v_presser_attacker_next = v_presser_attacker_next/np.linalg.norm(v_presser_attacker_next)
                            towards_attacker = np.dot(v_presser_attacker_next, dir_presser) > np.cos(pressure_params['angle_towards']*np.pi/180)
                            
                        ## Pull player position for final distance check
                        distance_p2p_past = np.sqrt((all_players_past[presser_id]['x']-attacker_past_xy[0])**2+(all_players_past[presser_id]['y'] - attacker_past_xy[1])**2)

                        ## Check if presser is moving closer to the attacker, is within sight/presence of attacker and is moving towards the attacker
                        if distance_p2p < distance_p2p_past and within_radius and towards_attacker:
                            
                            continuous.append(presser_id)
                            distances.append(round(distance_p2p,2))
      
        if len(continuous) > 0:
            possession.at[current_event.Index,'pressure'] = ';'.join(list(set(continuous)))
            possession.at[current_event.Index,'pressure_distance'] = ';'.join([str(d) for d in distances])
        
    possession.reset_index(inplace=True, drop=True)
    
    continuous_pressure = possession.set_index('frame')['pressure'].to_dict()
    model['pressure'] = model['frame'].apply(lambda x: continuous_pressure.get(x,''))
    pressure_distance = possession.set_index('frame')['pressure_distance'].to_dict()
    model['pressure_distance'] = model['frame'].apply(lambda x: pressure_distance.get(x,''))

    team_possession, team_pressures = create_team_pressure(model, team_possession, params)

    return model, team_possession, team_pressures


def create_team_pressure(model, team_possession, params):

    """
    This routine collapses individual pressures into team pressures that encompass frames where no player has possession as well 
    (as long as they are shorter than `max_team_press_cutoff`), and merges them in the `team_possession` dataframe. 
    It also creates a `team_pressures` dataframe with start/end time of each team pressure   

    :param model: Events dataframe
    :param team_possession: Team possession dataframe
    :param params: Dictionary containing all the relevant parameters    
    :return: Team possession dataframe updated, team pressures dataframe
    """     

    pressures = model.query("pressure != ''").copy().reset_index(drop=True)
    team_possession['pressure'] = ''
    team_possession.set_index('time_ms', inplace=True)
    home_id = params['home_id']
    away_id = params['away_id']
    other_team = {home_id:away_id, away_id:home_id}
    
    team_pressures = []
    for press in pressures.itertuples():
        if press.Index == 0:
            start_time = press.time_ms
            end_time = press.time_ms
            this_team = press.team_possession
            continue
        
        no_oop = (team_possession.loc[end_time:press.time_ms,'ball_status'] == 'in_play').all()     
        no_interruptions = (team_possession.loc[end_time:press.time_ms,'is_home'].diff().dropna() == 0).all() 

        if this_team == press.team_possession and press.time_ms - end_time < params['pressure']['max_team_press_cutoff']*1000 and no_interruptions and no_oop:
            end_time = press.time_ms
        else:
            team_possession.loc[start_time:end_time,'pressure'] = other_team[this_team]
            team_pressures.append([start_time, end_time, other_team[this_team]])
            start_time = press.time_ms
            end_time = press.time_ms
            this_team = press.team_possession
    
    team_possession.loc[start_time:end_time,'pressure'] = other_team[this_team]    
    team_pressures.append([start_time, end_time, other_team[this_team]])
    team_pressures = pd.DataFrame(team_pressures, columns=['start_time_ms', 'end_time_ms', 'team_pressing'])
    team_possession.reset_index(inplace=True)
    
    return team_possession, team_pressures
        

def set_team_possession(model, params):

    """
    This routine team possession at every frame of the tracking data, not only the frames that have events associated. Team possession
    is established based on the possession information from the events dataframe, but looks at the temporal logic whereby if a player
    gains possession for less than X seconds or less than X passes completed, possession still remains with the other team.


    :param model: Events dataframe
    :param params: Dictionary containing all the relevant parameters    
    :return: Team possession dataframe
    """ 

    home_id = params['home_id']
    away_id = params['away_id']
    home_name = params['home_team_name']
    away_name = params['away_team_name']


    team_ids = [home_id, away_id]
    id_to_name = {home_id:home_name, away_id:away_name}
    other_team = {home_id:away_id, away_id:home_id}
    time_thresh = params['possession']['time_thresh']
    pass_thresh = params['possession']['pass_thresh']
    set_piece_thresh = params['possession']['set_piece_thresh']

    fps = params['fps']

    ## Establish changes in possession from the events dataframe, accounting for the time, passes and set pieces constraint
    df = model[['frame','event_time','time_ms','ball_control','team_id',
                                        'event_name','from_set_piece','half']].query("ball_control != 'no control'").reset_index(drop=True).copy()
    half = df.iloc[0]['half']
    df['frame'] = df['frame'].astype('int64')
    team_attacking = df.iloc[0]['team_id'].split(';')[0]  
    first_possession = df.iloc[0]['frame']
    prev_change = df.iloc[0]['frame']

    df['team_id'] = df['team_id'].str.split(';').str[0]
    df['is_home'] = df['team_id'].map({home_id:1,
                                       away_id:0})
    df['change'] = df['is_home'].diff()
    df.at[0,'change'] = 0
    df.set_index('frame', inplace=True)

    change_frame = list(set(df.query("change != 0").index.tolist() + [params['td'].iloc[-1]['frame']]))
    change_frame.sort()
    df['attacking_team_id'] = ''
    for i,frame in enumerate(change_frame[:-1]):
        df.loc[prev_change:frame-1,'attacking_team_id'] = team_attacking
        next_frame = change_frame[i+1]
        interval = df.loc[frame:next_frame-1].copy()    
        interval['delta_t'] = interval['time_ms']-interval.iloc[0]['time_ms']
        
        frames_in_possession = interval.query("ball_control == 'possession'").shape[0]
        
        this_set_piece = (interval.query("delta_t <= 1000*@set_piece_thresh")['from_set_piece'] != '').any()
        time_constraint = frames_in_possession/fps >= time_thresh
        pass_constraint = interval['event_name'].str.contains('|'.join(['pass','cross','shot'])).sum() >= pass_thresh
        
        if time_constraint or pass_constraint or this_set_piece:
            team_attacking = df.at[frame,'team_id'].split(';')[0]         
   
        prev_change = frame
    
    df.loc[prev_change:,'attacking_team_id'] = team_attacking 
    df['attacking_name'] = df['attacking_team_id'].apply(lambda x: id_to_name[x])
    df_dict = df[['attacking_team_id']].to_dict(orient='index')
    df_dict = {k:v['attacking_team_id'] for k,v in df_dict.items()}

    ## Impute those changes in possession on the tracking data. Team possession before set pieces are taken is relevant, as we
    ## need to make sure that possession is switched to the team that will resume the game right when the ball goes OOP
    team_possession = params['td'][['frame','ball_status',
                                           'time_ms','half']].drop_duplicates(subset=['frame']).reset_index(drop=True).copy()

    ball_dead = team_possession[pd.eval("team_possession.ball_status=='dead'")].reset_index()
    dead_ball_intervals = []
    for k, segment in ball_dead.groupby(ball_dead['index'] - np.arange(ball_dead.shape[0])):

        f0 = segment.iloc[0]['frame']
        f1 = segment.iloc[-1]['frame']
        dead_ball_intervals.append([f0,f1])

    non_null_frames = df.query("change != 0").index.tolist()
    team_possession.set_index('frame', inplace=True)
    team_possession['attacking_team_id'] = ''
    team_possession.loc[:non_null_frames[0],'attacking_team_id'] = df.iloc[0]['attacking_team_id']

    t0 = time()
    for i,frame in enumerate(non_null_frames):
        oob_frame = frame
        
        try:
            interval_set_piece = df.loc[frame:non_null_frames[i+1]-1].copy()
        except:
            interval_set_piece = df.loc[frame:].copy()
        interval_set_piece['delta_t'] = interval_set_piece['time_ms']-interval_set_piece.iloc[0]['time_ms']    
        
            
        any_set_piece = (interval_set_piece.query("delta_t <= 1000*@set_piece_thresh")['from_set_piece'] != '').any()
        no_dead = False
        if any_set_piece:
            set_piece_frame = (interval_set_piece.query("delta_t <= 1000*@set_piece_thresh")['from_set_piece'] != '').idxmax()
            no_dead = (team_possession.loc[frame:set_piece_frame,'ball_status'] == 'in_play').all()

        if any_set_piece and no_dead:
            id_dbs = [True if frame > ni[1] and frame < dead_ball_intervals[i+1][0] else False for i,ni in enumerate(dead_ball_intervals[:-1])]
            if any(id_dbs):
                oob_frame = dead_ball_intervals[id_dbs.index(True)][0]            
        team_possession.loc[oob_frame:,'attacking_team_id'] = df_dict[frame]

    team_possession.reset_index(inplace=True)
    df.reset_index(inplace=True)
    team_possession['defending_team_id'] = team_possession['attacking_team_id'].map(other_team)
    team_possession['is_home'] = team_possession['attacking_team_id'].map({home_id:1,
                                                        away_id:0})
    team_possession['minute'] = team_possession['time_ms'] // 60000
    team_possession['sec'] = (team_possession['time_ms']/1000) % 60

    return team_possession[['frame','time_ms','minute','sec','attacking_team_id','defending_team_id','is_home','half','ball_status']].merge(df[['frame','from_set_piece']], on='frame', how='left').fillna('')




###################################################################################################
############################################ POSSESSION MODEL #####################################
###################################################################################################

def assign_possession_contest(params):

    """
    This routine is the main function to assign which player(s) were closest to the ball at every frame, as well as contest information

    :param params: Dictionary containing all the relevant parameters    
    :return: Updated dictionary containing all the data generated, including possession data and updated ball data with next-frame information
    """ 

    td = params['td']
    ball_data = params['ball_data']
    hyperparams = params['hyperparams']

    home_id = params['home_id']
    away_id = params['away_id']

    frames = ball_data['frame'].tolist()
    xy_ball = ball_data[['ball_x','ball_y','ball_z']].to_numpy()
    time_data = ball_data['game_time'].tolist()
    half = ball_data.at[0,'half']
    ball_status = ball_data['ball_status'].tolist()
    

    ball_speed_vec = ball_data['ball_speed'].tolist()
    ball_displacement_vec = ball_data['ball_displacement'].tolist()
    xy_ball = ball_data[['ball_x','ball_y','ball_z']].to_numpy()
  

    ## Evaluate in-game frames for player proximity to ball. Establish players in possession zone and players in contest zone
    in_game = td[pd.eval("td.ball_status !=' dead'")].reset_index(drop=True)
    pz_contest = hyperparams['pz_contest']
    pz = hyperparams['pz']
    players_in_pz_contest_long = in_game[pd.eval("in_game.distance <= pz_contest")][['distance','player_id','team_id','x','y','frame']].sort_values(by=['frame','distance'], ascending=True)
    players_in_pz_contest = []
    cols_to_groupby = ['x','y','player_id','team_id','distance']
    for col in cols_to_groupby:
        players_in_pz_contest.append(players_in_pz_contest_long.groupby('frame')[col].apply(list))
    players_in_pz_contest = pd.concat(players_in_pz_contest, axis=1).reset_index().merge(pd.DataFrame(data=frames,
                                                                                      columns=['frame']),
                                                                         on='frame', how='right').set_index('frame').to_dict(orient='index')
    players_in_pz_contest = {key:(val if type(val['x']) == list else []) for key,val in players_in_pz_contest.items()}  
    players_in_pz_contest_long['pz'] = players_in_pz_contest_long['player_id'].apply(lambda x: hyperparams['pz'] if x in params['goalkeeper_ids'] else hyperparams['pz_gk'])    
    if pz < pz_contest:
        players_in_pz_long = players_in_pz_contest_long[pd.eval("players_in_pz_contest_long.distance <= players_in_pz_contest_long.pz")][['distance','player_id','team_id','x','y','frame']].sort_values(by=['frame','distance'], ascending=True)
        players_in_pz = []
        for col in cols_to_groupby:
            players_in_pz.append(players_in_pz_long.groupby('frame')[col].apply(list))
        players_in_pz = pd.concat(players_in_pz, axis=1).reset_index().merge(pd.DataFrame(data=frames,
                                                                                          columns=['frame']), on='frame', how='right').set_index('frame').to_dict(orient='index')
        players_in_pz = {key:(val if type(val['x']) == list else []) for key,val in players_in_pz.items()}

    else:
        players_in_pz = players_in_pz_contest.copy()
            
    ## Initialize variables before looping through all the frames. 0 refers to the current frame and 1 refers to the next frame
    ball_dir_1 = (np.nan,np.nan)
    ball_speed_1 = np.nan
    ball_x1, ball_y1, ball_z1 = xy_ball[0,:]
    players_in_pz_1 = players_in_pz.get(frames[0], [])
    players_in_pz_contest_1 = players_in_pz_contest.get(frames[0], [])
    ball_change_dir_vec = []
    ball_change_speed_vec = []
    ball_dir0_vec = []
    ball_dir1_vec = []
    ball_speed0_vec = []
    ball_speed1_vec = []
    model = []

    for index,frame in enumerate(frames[:-1]):

        possession_assigned = False
        contest_assigned = False

        ball_dir_0 = copy(ball_dir_1)
        ball_speed_0 = copy(ball_speed_1)
        ball_x0 = copy(ball_x1)
        ball_y0 = copy(ball_y1)
        ball_z0 = copy(ball_z1)

        players_in_pz_0 = copy(players_in_pz_1)
        players_in_pz_contest_0 = copy(players_in_pz_contest_1)

        time0 = time_data[index]
        timestamp = [time0, half, frame, ball_status[index], (ball_x0, ball_y0, ball_z0), ball_speed_0]

        ball_x1, ball_y1, ball_z1 = xy_ball[index+1,:]
        players_in_pz_1 = players_in_pz.get(frames[index+1], [])
        players_in_pz_contest_1 = players_in_pz_contest.get(frames[index+1], [])


        if np.isnan(ball_x1) or np.isnan(ball_x0):
            ball_displacement = np.nan
            ball_speed_1 = np.nan
            ball_dir_1 = (np.nan,np.nan)
            change_direction = np.nan
            change_speed = np.nan
        else:
            dx_ = ball_x1-ball_x0
            dy_ = ball_y1-ball_y0
            ball_displacement = ball_displacement_vec[index+1]
            
            ball_dir_1 = ball_direction((ball_x0, ball_y0),
                                        (ball_x1, ball_y1))
            ball_speed_1 = ball_speed_vec[index]
            
            if ball_dir_1[0] != 0 and ball_dir_1[1] != 0:
                change_direction = ball_dir_1[0]*ball_dir_0[0] + ball_dir_1[1]*ball_dir_0[1]
            else:
                change_direction = 1
            change_speed = ball_speed_1-ball_speed_0

        ball_change_dir_vec.append(change_direction)
        ball_change_speed_vec.append(change_speed)
        ball_dir0_vec.append(ball_dir_0)
        ball_dir1_vec.append(ball_dir_1)
        ball_speed0_vec.append(ball_speed_0)
        ball_speed1_vec.append(ball_speed_1)

        ## If ball is labeled as dead, we record it and move to the next frame
        if ball_status[index] == 'dead':
            action = timestamp + ['ball dead'] + 15*[np.nan] 
            model.append(action)
            del action
            continue

        action_contest_ = 5*['']
        action_possession_ = 4*['']  
        
        ## There is at least one player that has the ball within their possession zone
        if len(players_in_pz_0) > 0:
            closest_player = players_in_pz_0['player_id'][0]
            closest_player_team = players_in_pz_0['team_id'][0]

            same_team_pz = all(team == closest_player_team for team in players_in_pz_0['team_id'])

            if len(players_in_pz_1) > 0:
                next_frame_players = players_in_pz_1['player_id']
            else:
                next_frame_players = []

            ## Only one player with ball in PZ
            if len(players_in_pz_0['x']) == 1:
                action_possession_ = [closest_player, closest_player_team, str(players_in_pz_0['distance'][0]),
                         str((players_in_pz_0['x'][0],players_in_pz_0['y'][0]))]
                if closest_player in next_frame_players:      
                    action = ['possession'] + action_possession_ + action_contest_
                    possession_assigned = True
                else:
                    action = ['potential loss'] + action_possession_ + action_contest_
                    possession_assigned = True

            ## More than one player with ball in PZ
            elif len(players_in_pz_0['x']) > 1:
                action_possession_ = [';'.join(players_in_pz_0['player_id']),
                            ';'.join(players_in_pz_0['team_id']),
                            ';'.join([str(s) for s in players_in_pz_0['distance']]),
                            ';'.join([str((x,players_in_pz_0['y'][i])) for i,x in enumerate(players_in_pz_0['x'])])]  

                if same_team_pz:

                    if closest_player in next_frame_players:
                        action = ['possession'] + action_possession_ + action_contest_
                        possession_assigned = True
                    else:
                        action = ['potential loss'] + action_possession_ + action_contest_
                        possession_assigned = True

                elif not same_team_pz:
                    if closest_player not in next_frame_players:
                        action = ['potential loss'] + action_possession_ + action_contest_
                        possession_assigned = True
        else:
            continue


        ## There is a contest if at least two players from different times are within the pz_contest area of the ball
        if len(players_in_pz_contest_0) == 0:
            contest_assigned = False
        else:
            contest_assigned = (len(players_in_pz_contest_0['x'])>1) and (len(set(players_in_pz_contest_0['team_id'])) > 1)
        
        if contest_assigned:
            closest_player_team = players_in_pz_contest_0['team_id'][0]
            i_player = ([team != closest_player_team for team in players_in_pz_contest_0['team_id']]).index(True) 
            player_distance = np.sqrt((players_in_pz_contest_0['x'][i_player] - players_in_pz_contest_0['x'][0])**2 +
                                  (players_in_pz_contest_0['y'][i_player] - players_in_pz_contest_0['y'][0])**2)
            contest_ids = players_in_pz_contest_0['player_id'][0] + ';' + players_in_pz_contest_0['player_id'][i_player]

            team_ids = closest_player_team + ';' + home_id if closest_player_team != home_id else closest_player_team + ';' + away_id
            contest_distance = str(players_in_pz_contest_0['distance'][0]) + ';' + str(players_in_pz_contest_0['distance'][i_player])
            player_xy = f"{(players_in_pz_contest_0['x'][0],players_in_pz_contest_0['y'][0])};" + f"{(players_in_pz_contest_0['x'][i_player],players_in_pz_contest_0['y'][i_player])}"

            action_contest_ = ['contest',
                            contest_ids,
                            team_ids,
                            contest_distance,
                            player_xy]
            if possession_assigned == True:
                if action[0] =='potential loss':
                    action = ['potential loss'] + action_possession_ + action_contest_
                else:
                    action = ['possession'] + action_possession_ + action_contest_
            else:
                action = ['no control'] + action_possession_ + action_contest_

        del action_possession_, action_contest_

        if possession_assigned or contest_assigned:
            model.append(timestamp + action + ['0',ball_displacement, change_direction,
                                              change_speed,
                                               ball_dir_0,ball_dir_1])  
            del action


    raw_possession_model = pd.DataFrame(model, columns=['event_time','half','frame',
                                                    'ball_status', 'ball_pos','ball_speed','event_name',
                                                    'player_id','team_id','distance','xy', 
                                                    'contest','contest_player_id','contest_team_id','contest_distance','contest_xy',
                                    'step_change','ball_displacement','change_in_direction',
                                                    'change_in_speed','ball_dir_0', 'ball_dir_1'])
    raw_possession_model['total_frames'] = len(frames)

    ball_change_dir_vec.append(np.nan)
    ball_change_speed_vec.append(np.nan)
    ball_dir0_vec.append(ball_dir_1)
    ball_dir1_vec.append((np.nan,np.nan))
    ball_speed0_vec.append(ball_speed_1)
    ball_speed1_vec.append(np.nan)

    ## Incorporate current and future velocity & direction to ball_data
    ball_data['change_direction'] = ball_change_dir_vec
    ball_data['change_speed'] = ball_change_speed_vec
    ball_data['direction_0'] = ball_dir0_vec
    ball_data['direction_1'] = ball_dir1_vec
    ball_data['speed_0'] = ball_speed0_vec
    ball_data['speed_1'] = ball_speed1_vec  

    return raw_possession_model, ball_data


def possession_model(data_input):
  
    """
    This routine is the main function to generate possession information from raw tracking data, as well as set piece information.

    :param data_input: Dictionary containing only the relevant parameters, including tracking data
    :return: Updated dictionary containing all the data generated, including possession data, dead ball triggers, log_incidence, frame count per player
    """ 

    params = data_input['params']
    model = {}
    raw_possession_model, ball_data = assign_possession_contest(params)
    params['ball_data'] = ball_data
    model['raw_possession_model'] = raw_possession_model
    model['frame_count'] = params['td'][['frame','player_id',
                        'ball_status']].groupby(['player_id',
                                                 'ball_status']).agg('count').rename(columns={'frame':'frame_count'})

    log_incidence = dict()
    dead_ball_output, log_incidence = extract_dead_ball_segments(params, log_incidence)
    model.update(dead_ball_output) 
    

    ## Create the return dictionary
    data_output = {}
    data_output['params'] = params
    data_output['models'] = model
    data_output['log_incidence'] = log_incidence
    return data_output



def goalkick_trigger(player_xy, active_goal_x, params):

    """
    This routine checks whether the players on a given frame satisfy the goal-kick trigger conditions

    :param player_xy: Coordinates of players in a given frame as a numpy array
    :param active_goal_x: x-coordinate of goal where players are attacking towards
    :param params: Dictionary containing all the relevant parameters    
    :return: True if player is in the goal area (with a buffer tol_gk)
    """ 

    smbx_dim = params['pitch_measures']['smbx_dim']
    goal_dim = params['pitch_measures']['goal_dim']

    if active_goal_x > 0:
        return player_xy[0] <= -params['pitch_length']/2 + smbx_dim + params.get('tol_gk',0) and abs(player_xy[1]) <= smbx_dim+goal_dim/2 + params.get('tol_gk',0)
    else:
        return player_xy[0] >= params['pitch_length']/2 -smbx_dim - params.get('tol_gk',0) and abs(player_xy[1]) <= smbx_dim+goal_dim/2+ params.get('tol_gk',0)


def throwin_trigger(player_xy, params):

    """
    This routine checks whether the players on a given frame satisfy the throw-in trigger conditions

    :param player_xy: Coordinates of players in a given frame as a numpy array
    :param params: Dictionary containing all the relevant parameters    
    :return: Signed distance of player to the y-coordinate pitch_width/2-tol_ti. Negative values if above this line
    """ 

    try:
        return -abs(player_xy[:,1]) + (params['pitch_width']/2 - params.get('tol_ti',0))
    except:
        return -abs(player_xy[1]) + (params['pitch_width']/2 - params.get('tol_ti',0))



def corner_trigger(player_xy, active_goal_x, corner_spots):
  
    """
    This routine checks whether the players on a given frame satisfy the corner trigger conditions

    :param player_xy: Coordinates of players in a given frame as a numpy array
    :param active_goal_x: x-coordinate of goal where players are attacking towards
    :param corner_spots: x-y coordinates of all four corner spots
    :return: Minimum distance of each input player to any of the four corner spots
    """ 

    try:
        return np.minimum(np.sqrt((player_xy[0]-active_goal_x*corner_spots[0][0])**2 + (player_xy[1]-corner_spots[0][1])**2),
                         np.sqrt((player_xy[0]-active_goal_x*corner_spots[1][0])**2 + (player_xy[1]-corner_spots[1][1])**2))
    except:
        return np.minimum(np.sqrt((player_xy[:,0]-active_goal_x*corner_spots[0][0])**2 + (player_xy[:,1]-corner_spots[0][1])**2),
                         np.sqrt((player_xy[:,0]-active_goal_x*corner_spots[1][0])**2 + (player_xy[:,1]-corner_spots[1][1])**2))



def penalty_trigger(frame_players, params):
    """
    This routine checks whether the players on a given frame satisfy the penalty trigger conditions

    :param frame_players: Coordinates, team ID and player ID of all players in a given frame
    :param params: Dictionary containing all the relevant parameters    
    :return: The ID of the single player that is in the opposing penalty kick box, or NaN if none is found
    """ 

    home_id = params['home_id']
    away_id = params['away_id']
    home_players = frame_players.query("team_id == @home_id").copy().reset_index(drop=True)
    away_players = frame_players.query("team_id == @away_id").copy().reset_index(drop=True)


    home_players['in_opp_pk_box'] = penalty_shooter(home_players.x.values, home_players.y.values,
                                                 home_players.active_goal, params)

    ## penalty in away area
    if abs(home_players['in_opp_pk_box'].sum()) == 1:
        frame_players['between_goal_pkspot'] = frame_players.apply(lambda row: between_goal_pkspot(row['x'], row['y'], np.sign(params['active_goal'][home_id][0]), params), axis=1)
        if frame_players['between_goal_pkspot'].sum() == 2:
            away_players['in_goal_line'] = away_players.apply(lambda row: in_own_goal_line(row['x'], row['y'],
                                                                                     row['active_goal'], params), axis=1)
            if abs(away_players['in_goal_line'].sum()) == 1:
                return home_players.at[home_players['in_opp_pk_box'].idxmax(),'player_id']


    away_players['in_opp_pk_box'] = penalty_shooter(away_players.x.values, away_players.y.values,
                                                 away_players.active_goal, params)
    ## penalty in home area
    if abs(away_players['in_opp_pk_box'].sum()) == 1:
        frame_players['between_goal_pkspot'] = frame_players.apply(lambda row: between_goal_pkspot(row['x'], row['y'], np.sign(params['active_goal'][away_id][0]), params), axis=1)
        if frame_players['between_goal_pkspot'].sum() == 2:
            home_players['in_goal_line'] = home_players.apply(lambda row: in_own_goal_line(row['x'], row['y'], row['active_goal'], params), axis=1)
            if abs(home_players['in_goal_line'].sum()) == 1:
                return away_players.at[away_players['in_opp_pk_box'].idxmax(),'player_id']

    return np.nan


def between_goal_pkspot(x, y, active_goal_x, params):

    """
    This routine checks if a player located in x-y is inside the area reserved only for the PK taker and the goalkeeper, with a tolerance tol_pk3

    :param x: x-coordinate of player
    :param y: y-coordinate of player
    :param active_goal_x: x-coordinate of goal where player is attacking towards
    :param params: Dictionary containing all the relevant parameters    
    :return: True if the player is between the goal and the PK spot
    """ 

    penspot_dim = params['pitch_measures']['penspot_dim']
    penbox_len = params['pitch_measures']['penbox_len']
    penbox_wid = params['pitch_measures']['penbox_wid']
    goal_height = params['pitch_measures']['goal_height']
    arc_dim = params['pitch_measures']['arc_dim']
    y_circle = np.sin(53*np.pi/180)*arc_dim

    if active_goal_x > 0:
        xp = params['pitch_length']/2 - penspot_dim + params.get('tol_pk3',0)
        xa = params['pitch_length']/2 - penbox_len + params.get('tol_pk3',0)
        off_circle = x < params['pitch_length']/2 + goal_height and x > xa and abs(y) < penbox_wid/2 - params.get('tol_pk3',0) and abs(y) >= y_circle
        in_circle = x < params['pitch_length']/2 + goal_height and ((x-xp)**2 + y**2 < arc_dim**2 or x > xa) and abs(y) < y_circle    
    else:
        xp = -params['pitch_length']/2 + penspot_dim - params.get('tol_pk3',0)
        xa = -params['pitch_length']/2 + penbox_len - params.get('tol_pk3',0)
        off_circle = x > -params['pitch_length']/2 - goal_height and x < xa and abs(y) < penbox_wid/2 - params.get('tol_pk3',0) and abs(y) >= y_circle
        in_circle = x > -params['pitch_length']/2 - goal_height and ((x-xp)**2 + y**2 < arc_dim**2 or x < xa) and abs(y) < y_circle   
    return off_circle or in_circle


def penalty_shooter(x, y, active_goal_x, params):
  
    """
    This routine checks if a player located in x-y is located in the penalty shooter box, with a tolerance tol_pk2

    :param x: x-coordinate of player
    :param y: y-coordinate of player
    :param active_goal_x: x-coordinate of goal where player is attacking towards
    :param params: Dictionary containing all the relevant parameters    
    :return: True if the player is in the penalty shooter box
    """ 

    return np.logical_and(abs(y) <= params.get('tol_pk2',0),
                          abs(x - active_goal_x*(params['pitch_length']/2 - params['pitch_measures']['penspot_dim'] - params.get('tol_pk2',0)/4)) <= 3*params.get('tol_pk2',0)/4)

def in_own_goal_line(x, y, active_goal_x, params):
    """
    This routine checks if a player located in x-y is located in their own goal line, with a tolerance tol_pk1

    :param x: x-coordinate of player
    :param y: y-coordinate of player
    :param active_goal_x: x-coordinate of goal where player is attacking towards
    :param params: Dictionary containing all the relevant parameters    
    :return: True if the player is in their own goal line
    """ 

    return np.logical_and(abs(x + active_goal_x*(params['pitch_length']/2)) <= params.get('tol_pk1',0),
                          abs(y) <= params['pitch_measures']['goal_dim']/2 + params.get('tol_pk1',0))

    

def extract_penalty_trigger(params):
    
    """
    This routine searches for the frames in the tracking data where a penalty configuration is present 

    :param params: Dictionary containing all the relevant parameters        
    :return: Dataframe containing the frames where with penalty configuration, along with the ID of the player in the penalty mark
    """ 

    ## First filter the tracking data to focus only on frames where there is at most one player in the penalty mark at least one goalkeeper in the goal line
    initial_filter = params['td'].copy()
    initial_filter['shooter_in_pos'] = penalty_shooter(initial_filter['x'].values,
                                                    initial_filter['y'].values, initial_filter['active_goal'].values,
                                                    params)

    initial_filter['goalie_in_line'] = in_own_goal_line(initial_filter['x'].values,
                                                    initial_filter['y'].values, initial_filter['active_goal'].values,
                                                    params)

    goalie_in_line = np.logical_or(initial_filter.groupby('frame')['goalie_in_line'].agg(sum) == 1,
                               initial_filter.groupby('frame')['goalie_in_line'].agg(sum) == 2)
    potential_penalty = np.logical_and(initial_filter.groupby('frame')['shooter_in_pos'].agg(sum) == 1,
                                   goalie_in_line)
    potential_penalty_frames = potential_penalty[potential_penalty].index.tolist()

    ## For these frames, we investigate the location of the remaining players to determine if there is a penalty configuration
    penalty_frames_df = initial_filter[pd.eval("initial_filter.frame.isin(potential_penalty_frames)")].reset_index(drop=True)
    penalty_frames_df = penalty_frames_df.groupby('frame')[['x','y','team_id','player_id','distance','active_goal']].apply(lambda x: penalty_trigger(x, params)).dropna().reset_index()
    if penalty_frames_df.empty:
        penalty_frames_df = pd.DataFrame(columns=['frame','player_id','x','y','trigger'])
    else:
        penalty_frames_df.rename(columns={0:'player_id'}, inplace=True)
        penalty_frames_df = penalty_frames_df.merge(params['td'][['frame','player_id','x','y']], on=['frame','player_id'], how='left')
        penalty_frames_df['trigger'] = 0
    

    return penalty_frames_df.merge(params['ball_data'][['frame','game_time','ball_status']], on='frame')


def kickoff_average_y_10yards(player_y, params):
    """
    This routine checks whether the average y-coordinate of all players falls within the center arc. 
    This is used as a heuristic for differentiating real kickoffs from VAR situations where players are elsewhere awaiting a decision


    :param player_y: y-coordinate of players in a given frame
    :param params: Dictionary containing all the relevant parameters            
    :return: True if the average y-coordinate of players is less than the center arc radius
    """ 

    
    if abs(player_y.mean()) < params['pitch_measures']['arc_dim']:
        return True
    else:
        return False     


def extract_kickoff_trigger(params):
    """
    This routine searches for the frames in the tracking data where a kickoff configuration is present 

    :param params: Dictionary containing all the relevant parameters        
    :return: Dataframe containing the frames where with kickoff configuration, along with the ID of the player in the center mark
    """ 

    home_id = params['home_id']
    away_id = params['away_id']
    td = params['td']
    frames = params['ball_data']['frame'].tolist()

    ## First examine the frames where all players are in their respective halves of the pitch
    home_active_goal = params['active_goal'][home_id][0]
    away_active_goal = params['active_goal'][away_id][0]  

    home_x = td[pd.eval("td.team_id == home_id")][['x','frame']].groupby('frame', sort=False).agg('median')['x'].tolist()
    away_x = td[pd.eval("td.team_id == away_id")][['x','frame']].groupby('frame', sort=False).agg('median')['x'].tolist()
    if home_active_goal > away_active_goal:
        possible_kickoff = [frame for index,frame in enumerate(frames) if home_x[index] < params.get('tol_ko1',0) and away_x[index] > -params.get('tol_ko1',0)]
        inpitch = td[pd.eval("td.frame.isin(possible_kickoff) and abs(td.x) < params['pitch_length']/2-params.get('kickoff_player_out_pitch',0) and abs(td.y) < params['pitch_width']/2-params.get('kickoff_player_out_pitch',0)")].copy()
        home_in_own = inpitch[pd.eval("inpitch.team_id == home_id")][['x','frame']].groupby('frame', sort=False).agg(lambda x_pos: (x_pos < params.get('tol_ko1',0)).sum() == x_pos.count())
        away_in_own = inpitch[pd.eval("inpitch.team_id == away_id")][['x','frame']].groupby('frame', sort=False).agg(lambda x_pos: (x_pos > -params.get('tol_ko1',0)).sum() == x_pos.count())
    else:
        possible_kickoff = [frame for index,frame in enumerate(frames) if home_x[index] > -params.get('tol_ko1',0) and away_x[index] < params.get('tol_ko1',0)]
        inpitch = td[pd.eval("td.frame.isin(possible_kickoff) and abs(td.x) < params['pitch_length']/2-params.get('kickoff_player_out_pitch',0) and abs(td.y) < params['pitch_width']/2-params.get('kickoff_player_out_pitch',0)")].copy()
        away_in_own = inpitch[pd.eval("inpitch.team_id == away_id")][['x','frame']].groupby('frame', sort=False).agg(lambda x_pos: (x_pos < params.get('tol_ko1',0)).sum() == x_pos.count())
        home_in_own = inpitch[pd.eval("inpitch.team_id == home_id")][['x','frame']].groupby('frame', sort=False).agg(lambda x_pos: (x_pos > -params.get('tol_ko1',0)).sum() == x_pos.count())


    ## Then on the frames where both teams are in their halves of pitch, filter frames with players close to the center mark
    both_in_own = []    
    for index, frame in enumerate(frames):
        try:
            both_in_own.append(home_in_own.at[frame,'x'] and away_in_own.at[frame,'x'])
        except:
            both_in_own.append(False)

    own_half_frames = [frame for index, frame in enumerate(frames) if both_in_own[index]==True]
    own_half_frames_df = td[pd.eval("td.frame.isin(own_half_frames)")].copy()
    own_half_frames_df['trigger_center'] = np.sqrt(own_half_frames_df.x**2 + own_half_frames_df.y**2)
    potential_kickoff_frames_df = own_half_frames_df[pd.eval("own_half_frames_df.trigger_center <= params.get('tol_ko2',0)")].sort_values(by=['frame','trigger_center'], ascending=True).reset_index(drop=True).copy()
    potential_kickoff_frames_dict = dict(tuple(potential_kickoff_frames_df.groupby('frame')))
    kickoff_10y_frames = potential_kickoff_frames_df.groupby('frame')['y'].apply(lambda y: kickoff_average_y_10yards(y, params)).dropna().reset_index().rename(columns={'y':'trigger_10yard'}).set_index('frame')['trigger_10yard'].to_dict()

    ## Loop over potential kickoff frames and if the players in the center spot belong to the same team and 
    ## the average y coordinate of all players is within 10yards, add the frame as kickoff
    if len(potential_kickoff_frames_dict) > 0:
        kickoff_frames = []
        for frame, frame_players in potential_kickoff_frames_dict.items():
            if len(frame_players['team_id'].unique()) < 2 and kickoff_10y_frames[frame]:   
                kickoff_frames.append(frame_players[['time_ms','frame','ball_status','player_id','x','y','trigger_center']]) 
        kickoff_frames = pd.concat(kickoff_frames).rename(columns={'time_ms':'game_time', 'trigger_center':'trigger'})
    else:
        kickoff_frames = pd.DataFrame(columns=['game_time','frame','ball_status','player_id','x','y','trigger'])
    
    return kickoff_frames

def extract_corner_trigger(params):

    """
    This routine searches for the frames in the tracking data where a corner configuration is present 

    :param params: Dictionary containing all the relevant parameters        
    :return: Dataframe containing the frames where with corner configuration, along with the ID of the player in the corner mark
    """ 

    td = params['td']
    ball_data = params['ball_data']

    dead_ball = td[pd.eval("td.ball_status =='dead'")][['frame','x','y',
                                                        'player_id','active_goal','distance','ball_status']].copy()
    ylim = params['pitch_width']/2 - max(params.get('tol_ck',0),params.get('tol_ti',0))
    dead_ball['player_sideline'] = np.where(abs(dead_ball['y']) >= ylim, True, False)

    corner_spots = [(params['pitch_length']/2,params['pitch_width']/2),
                (params['pitch_length']/2,-params['pitch_width']/2)]

    potential_corner_df = dead_ball[pd.eval("dead_ball.player_sideline == True")].drop(columns=['player_sideline']).copy()
    potential_corner_df['trigger'] = corner_trigger(potential_corner_df[['x','y']].to_numpy(),
                                                    potential_corner_df['active_goal'].values, corner_spots)
    corner_frames_df = potential_corner_df[pd.eval("potential_corner_df.trigger <= params.get('tol_ck',0)")].reset_index(drop=True).copy()

    return corner_frames_df.merge(ball_data[['frame','game_time']], on='frame')


def extract_throwin_trigger(params):

    """
    This routine searches for the frames in the tracking data where a throw-in configuration is present 

    :param params: Dictionary containing all the relevant parameters        
    :return: Dataframe containing the frames where with throw-in configuration, along with the ID of the player in the sideline
    """ 

    td = params['td']

    dead_ball = td[pd.eval("td.ball_status =='dead'")][['frame','x','y',
                                                        'player_id','active_goal','distance','ball_status']].copy()
    ylim = params['pitch_width']/2 - max(params.get('tol_ck',0),params.get('tol_ti',0))
    dead_ball['player_sideline'] = np.where(abs(dead_ball['y']) >= ylim, True, False)
    potential_throwin_df = dead_ball[pd.eval("dead_ball.player_sideline == True")].drop(columns=['player_sideline']).copy()

    potential_throwin_df['trigger'] = throwin_trigger(potential_throwin_df[['x','y']].to_numpy(), params)
    throwin_frames_df = potential_throwin_df[pd.eval("potential_throwin_df.trigger <= 0")].reset_index(drop=True)
    
    return throwin_frames_df.merge(params['ball_data'][['frame','game_time']], on='frame')


def extract_goalkick_trigger(dead_ball_intervals, params):
    """
    This routine searches for the frames in the tracking data where a goal kick configuration is present 

    :param dead_ball_intervals: Dataframe containing one dead ball interval per row, along with potential set piece that resumes the play
    :param params: Dictionary containing all the relevant parameters        
    :return: Dataframe containing the frames where with goalkick configuration, along with the ID of the player executing the goal kick
    """ 

    goalkick = []
    td = params['td']
    for row in dead_ball_intervals.itertuples():
        next_frame = td[pd.eval("td.frame == row.frame1+1")].sort_values(by='distance', ascending=True).copy()
        goalies = next_frame[pd.eval("next_frame.player_id.isin(params['goalkeeper_ids'])")] 

        ## If the goalie closest to ball has activated trigger and is closer to ball than any other player and ball is not in goal,
        ## then assign goal kick. 
        if not goalies.empty and goalkick_trigger(goalies.iloc[0][['x','y']].to_numpy(), 
                                                  np.sign(params['active_goal'][goalies.iloc[0]['team_id']][0]), 
                                                  params):
            infielders = next_frame[pd.eval("~next_frame.player_id.isin(params['goalkeeper_ids'])")] 
            goalie_closer_infielders = goalies.iloc[0]['distance'] < infielders.iloc[0]['distance']  
            ball_in_goal = check_ball_in_goal_2d((goalies.iloc[0]['ball_x'], goalies.iloc[0]['ball_y']), params)
            if goalie_closer_infielders and not ball_in_goal:
                goalkick.append({'player_id':goalies.iloc[0]['player_id'],
                                  'player_pos':[(goalies.iloc[0]['x'], goalies.iloc[0]['y'])]})
                continue
        goalkick.append(False)

    return goalkick

def get_index_from_trigger(trigger):
    """
    This routine returns the start and end frame of when a player(s) is satisfying the trigger for a given set piece, together with their ID and
    coordinates at the end frame. Because there may be multiple instances when that is satisfied, the index_setpiece is a list of lists, where each 
    inner list encompasses a start/end frame of the player satisfying the trigger

    :param trigger: Dataframe generated in possession_model containing frame, player_id, coordinates and trigger distance (if any)
    :return: List of lists where each list has start frame, end frame, player ID and player coordinates that satisfy said trigger
    """ 
    index_setpiece = []
    if 'ball_status' in trigger.columns:
        setpiece_detector = trigger.query("ball_status == 'dead'")        
    else:
        setpiece_detector = trigger
    setpiece_frames = setpiece_detector[['frame']].drop_duplicates(subset=['frame']).reset_index(drop=True)
                                    

    if not setpiece_frames.empty:
        for k, segment in setpiece_frames.groupby(setpiece_frames['frame'] - np.arange(setpiece_frames.shape[0])): 
            f1 = segment.iloc[-1]['frame']
            tm = setpiece_detector[pd.eval("setpiece_detector.frame == f1")].sort_values(by='trigger', ascending=True)
            index_setpiece.append([segment.iloc[0]['frame'], f1,
                             ';'.join(tm['player_id'].tolist()),
                            list(zip(tm.x, tm.y))])

    return index_setpiece



def isin_complete_trigger(i0, i1, trigger):

    """
    This routine checks if a given dead ball trigger is complete, namely it lasts until the end of the dead ball interval

    :param i0: First frame
    :param i1: Last frame
    :param trigger: Trigger list as the output of get_index_from_trigger
    :return: Dictionary with the player ID and position on the last frame
    """ 

    if len(trigger) == 0:
        return False
    
    players = []
    positions = []
    for (k0,k1,pl,pos) in trigger:
        if k0 >= i0 and k0 <= i1 and k1 == i1:
            players.append(pl)
            positions.append(pos)
  
    if len(players) > 0:
        return {'player_id':players[-1],'player_pos':positions[-1]}
    else:
        return False

def isin_incomplete_trigger(i0, i1, trigger):

    """
    This routine checks if a given dead ball trigger is incomplete, namely does not last until the end of the dead ball interval

    :param i0: First frame
    :param i1: Last frame
    :param trigger: Trigger list as the output of get_index_from_trigger
    :return: Dictionary with the player ID, position and frame on the last frame the trigger is satisfied
    """ 

    if len(trigger) == 0:
        return False
    
    players = []
    frames = []
    positions = []
    for (k0,k1,pl,pos) in trigger:
        if k0 >= i0 and k0 <= i1 and k1 <= i1:
            players.append(pl)
            positions.append(pos)
            frames.append(k1)
  
    if len(players) > 0:
        return {'player_id':players[-1],'player_pos':positions[-1],'frame':frames[-1]}
    else:
        return False

def check_cornerkick(row, next_frame, params):

    """
    This routine checks if the players identified as potential corner takers are present in the next_frame (first in-play frame after dead segment),
    and are within `pz_setpiece` of the ball. If no players are found, we check for either throw-in or goal kick

    :param row: Row of the db_intervals dataframe
    :param next_frame: Snapshot of tracking data at first in-play frame
    :param params: Dictionary containing all the relevant parameters
    :return: List of set piece event, player ID, location and ball height to construct db_intervals
    """ 

    player_ids = row.corner['player_id'].split(';')
    corner_players = next_frame.loc[next_frame['distance'] < params['pz_setpiece']]
 
    ball_z = next_frame.iloc[0]['ball_z']

    if not corner_players.empty and corner_players.iloc[0]['player_id'] in player_ids:          
        return ['corner kick', corner_players.iloc[0]['player_id'],
            (corner_players.iloc[0]['x'], corner_players.iloc[0]['y']),
            ball_z, '0']
    else:
        if row.throwin != False:
            return check_throwin(row, next_frame, params)
        else:
            return check_goalkick(next_frame, params)

def check_throwin(row, next_frame, params):

    """
    This routine checks if the players identified as potential throw-in takers are present in the next_frame (first in-play frame after dead segment),
    and are within `pz_setpiece` of the ball. If no players are found, we check for goal kick

    :param row: Row of the db_intervals dataframe
    :param next_frame: Snapshot of tracking data at first in-play frame
    :param params: Dictionary containing all the relevant parameters
    :return: List of set piece event, player ID, location and ball height to construct db_intervals
    """ 

    player_ids = row.throwin['player_id'].split(';')
    throwin_players = next_frame.loc[next_frame['distance'] < params['pz_setpiece']]

    ball_z = next_frame.iloc[0]['ball_z']

    if not throwin_players.empty and throwin_players.iloc[0]['player_id'] in player_ids:              
        return ['throw-in', throwin_players.iloc[0]['player_id'],
            (throwin_players.iloc[0]['x'], throwin_players.iloc[0]['y']),
            ball_z, '0']
    else:
        return check_goalkick(next_frame, params)


def check_goalkick(next_frame, params):

    """
    This routine checks players (not only goalies) in the next_frame (first in-play frame after dead segment) 
    that are within `pz_setpiece` of the ball and satisfy the GK trigger. If the trigger is not satisfied, we 
    assign a free kick, but if there is no player nearby we label it as undefined. 

    :param next_frame: Snapshot of tracking data at first in-play frame
    :param params: Dictionary containing all the relevant parameters
    :return: List of set piece event, player ID, location and ball height to construct db_intervals
    """ 

    goalkick_players = next_frame.loc[(next_frame['distance'] < params['pz_setpiece'])]  
    ball_z = next_frame.iloc[0]['ball_z']
    if not goalkick_players.empty and goalkick_trigger(goalkick_players.iloc[0][['x','y']].to_numpy(),
                                                       np.sign(params['active_goal'][goalkick_players.iloc[0]['team_id']][0]), 
                                                       params):
        return ['goal kick', goalkick_players.iloc[0]['player_id'],
            (goalkick_players.iloc[0]['x'], goalkick_players.iloc[0]['y']),            
            ball_z, '0']
    else:
        not_goalkick = next_frame.loc[next_frame['distance'] < params['pz_setpiece']]
        if not not_goalkick.empty:
            return ['free kick', not_goalkick.iloc[0]['player_id'],
                    (not_goalkick.iloc[0]['x'], not_goalkick.iloc[0]['y']),                            
                    ball_z, '0']
        else:
            return ['undefined', '', (np.nan, np.nan), 0, '0']


def extract_dead_ball_segments(params, log_incidence):
  
    """
    This routine creates the dataframe db_intervals, containing for each dead ball interval the start/end time, whether each
    dead ball trigger has been found during that segment and the player that satisfied it, and the most likely set piece that follows. 
    Specific parts of the routine are commented for better explainability. The hierarchy for defining set piece events is thoroughly detailed
    in the auto-eventing paper doi.org/10.1007/s12283-022-00381-6. At the end we also create dictionaries to detect kickoff & penalty
    configurations while the game is in-play, with the objective of maximizing detection in the case of erroneous data.

    :param params: Dictionary containing all the relevant parameters
    :param log_incidence: Dictionary where the incidences when running the possession model are logged

    :return: Dictionary containing dead ball intervals and in-game kickoff/penalty frames, as well as log_incidence updated.
    """ 

    td = params['td'] 
    half = params['half']
    segment_id = params['segment_id']
    verbose = params['verbose']
    ball_data = params['ball_data'].copy()
    ball_dead = ball_data[pd.eval("ball_data.ball_status=='dead'")].reset_index()


    ## We first extract the frames from the tracking data where a set piece may occur using player configurations and set piece triggers
    ## KICKOFF TRIGGERS
    kickoff_trigger = extract_kickoff_trigger(params)
    
    ## PENALTY TRIGGERS
    penalty_trigger = extract_penalty_trigger(params)    

    ## THROW-IN TRIGGER
    throwin_trigger = extract_throwin_trigger(params)    

    ## CORNER TRIGGER
    corner_trigger = extract_corner_trigger(params)    
    

    ## We then check whether a complete trigger is activated on every dead ball interval
    index_kickoff = get_index_from_trigger(kickoff_trigger)
    index_pk = get_index_from_trigger(penalty_trigger)
    index_ck = get_index_from_trigger(corner_trigger)
    index_ti = get_index_from_trigger(throwin_trigger)
    
    dead_ball_intervals = []
    initial_frame = td.iloc[0]['frame']
    for k, segment in ball_dead.groupby(ball_dead['index'] - np.arange(ball_dead.shape[0])):
      
        f0 = segment.iloc[0]['frame']
        f1 = segment.iloc[-1]['frame']
        t0 = segment.iloc[0]['game_time']
        t1 = segment.iloc[-1]['game_time']
        if f1-f0 <= 1 or f0 <= initial_frame+params['fps']:
            continue
        dead_ball_intervals.append([f0,f1,t0,t1])

    dead_ball_intervals = pd.DataFrame(dead_ball_intervals, columns=['frame0','frame1','time0','time1'])
    dead_ball_intervals['kickoff'] = dead_ball_intervals.apply(lambda x: isin_complete_trigger(x['frame0'],x['frame1'], index_kickoff), axis=1)
    dead_ball_intervals['penalty'] = dead_ball_intervals.apply(lambda x: isin_complete_trigger(x['frame0'],x['frame1'], index_pk), axis=1)
    dead_ball_intervals['corner'] = dead_ball_intervals.apply(lambda x: isin_complete_trigger(x['frame0'],x['frame1'], index_ck), axis=1)
    dead_ball_intervals['throwin'] = dead_ball_intervals.apply(lambda x: isin_complete_trigger(x['frame0'],x['frame1'], index_ti), axis=1)

    ## We now incorporate the trigger for goalkicks, on this first iteration we restrict to only goalies
    dead_ball_intervals['goalkick'] = extract_goalkick_trigger(dead_ball_intervals, params)

    ## We then check whether an incomplete trigger is activated on every dead ball interval
    dead_ball_intervals['kickoff_incomplete'] = dead_ball_intervals.apply(lambda x: isin_incomplete_trigger(x['frame0'],x['frame1'], index_kickoff), axis=1)
    dead_ball_intervals['penalty_incomplete'] = dead_ball_intervals.apply(lambda x: isin_incomplete_trigger(x['frame0'],x['frame1'], index_pk), axis=1)
    dead_ball_intervals['corner_incomplete'] = dead_ball_intervals.apply(lambda x: isin_incomplete_trigger(x['frame0'],x['frame1'], index_ck), axis=1)
    dead_ball_intervals['throwin_incomplete'] = dead_ball_intervals.apply(lambda x: isin_incomplete_trigger(x['frame0'],x['frame1'], index_ti), axis=1)

    ## Set initial kickoff  
    i0 = ball_data[pd.eval("ball_data.ball_status!='dead'")].index[0] 
    half = ball_data.iloc[i0]['half']
    frame0 = ball_data.iloc[i0]['frame']
    first_frame = td[pd.eval("td.frame == frame0")].sort_values(by='distance', ascending=True)
    first_frame['distance_center'] = first_frame.apply(lambda x: np.sqrt(x['x']**2 + x['y']**2) < params.get('tol_ko2',0), axis=1)

    if not first_frame['distance_center'].any():
      
        msg = "No player at kickoff circle on first in-play frame, setting as kickoff player closest player to center"
        log_incidence[frame0] = msg
        if verbose:
            print(msg)

        time0 = ball_data.iloc[0]['game_time'] 
        frame0 = ball_data.iloc[0]['frame']
        first_frame = td[pd.eval("td.frame == frame0")].sort_values(by='distance', ascending=True)
        first_frame['distance_center'] = first_frame.apply(lambda x: np.sqrt(x['x']**2 + x['y']**2) < params.get('tol_ko2',0), axis=1)

    else:
        time0 = ball_data.iloc[i0]['game_time'] 
      
    kickoff_player = first_frame.at[first_frame['distance_center'].idxmax(),'player_id']
    kickoff_xy = first_frame.loc[first_frame['distance_center'].idxmax(),['x','y']].to_numpy()
    kickoff_dict = {'player_id':kickoff_player,'player_pos':(kickoff_xy[0],kickoff_xy[1])}
    initial_kickoff = pd.DataFrame([[frame0, frame0,
                       time0, time0, kickoff_dict,
                      False, False, False,False, kickoff_dict, False, False, False]], columns=dead_ball_intervals.columns)
    dead_ball_intervals = pd.concat([initial_kickoff, dead_ball_intervals], ignore_index=True)

    pattern = [['kickoff start', kickoff_player, (kickoff_xy[0],kickoff_xy[1]), ball_data.at[i0,'ball_z'], '0']]  


    ## Establish the set piece events based on hierarchy and the distance between set piece taker and the ball when play resumes
    for row in dead_ball_intervals.itertuples():
        if row.Index == 0:
          ## First set piece is initial kickoff that has been established
            continue
        next_f = row.frame1+1 
        next_frame = td[pd.eval("td.frame == next_f")].sort_values(by='distance', ascending=True)
        if next_frame.empty:
            pattern.append(['end of half','',(np.nan, np.nan),0,'0'])
            continue
        
        if next_frame['distance'].isnull().all():
            pattern.append(['no data','',(np.nan, np.nan),0,'0'])
            continue

        if row.kickoff != False:
            player_ids = row.kickoff['player_id'].split(';')
            kickoff_players = next_frame.loc[(next_frame['player_id'].isin(player_ids)) &
                                          (next_frame['distance'] < params['pz_setpiece'])]
            if not kickoff_players.empty:                     
                pattern.append(['kickoff', kickoff_players.iloc[0]['player_id'],
                            (kickoff_players.iloc[0]['x'],kickoff_players.iloc[0]['y']),
                            next_frame.iloc[0]['ball_z'],'0'])
            else:
                last_five_seconds = td[pd.eval("td.player_id in player_ids and td.frame >= row.frame1 - 5*params['fps'] and td.frame <= row.frame1 and td.distance < params['pz_setpiece'] and td.x**2 +td.y**2 <= params['tol_ko2']**2 and td.ball_x**2 +td.ball_y**2 <= params['tol_ko2']**2")].sort_values(by='time_ms', ascending=False)
                if not last_five_seconds.empty and (ball_data[pd.eval("ball_data.frame >= row.frame1 - 5*params['fps'] and ball_data.frame <= row.frame1")]['ball_displacement'] <= params['kickoff_ball_max_displacement']).all():
                    pattern.append(['kickoff', last_five_seconds.iloc[0]['player_id'],
                            (last_five_seconds.iloc[0]['x'],last_five_seconds.iloc[0]['y']),
                            last_five_seconds.iloc[0]['ball_z'],'0'])
                else:
                    not_kickoff = next_frame.loc[next_frame['distance'] < params['pz_setpiece']]
                    if not not_kickoff.empty:
                        pattern.append(['free kick', not_kickoff.iloc[0]['player_id'],
                                    (not_kickoff.iloc[0]['x'], not_kickoff.iloc[0]['y']),
                                    next_frame.iloc[0]['ball_z'], '0'])
                    else:
                        pattern.append(['undefined', '',(np.nan, np.nan), 0,'0'])
        else:
            if row.kickoff_incomplete != False:
                player_ids = row.kickoff_incomplete['player_id'].split(';')
                last_five_seconds = td[pd.eval("td.player_id in player_ids and td.frame >= row.kickoff_incomplete['frame'] - 5*params['fps'] and td.frame <= row.frame1 and td.distance < params['pz_setpiece'] and td.x**2 +td.y**2 <= params['tol_ko2']**2 and td.ball_x**2 +td.ball_y**2 <= params['tol_ko2']**2")].sort_values(by='time_ms', ascending=False)
                if not last_five_seconds.empty and (ball_data[pd.eval("ball_data.frame >= row.kickoff_incomplete['frame'] - 5*params['fps'] and ball_data.frame <= row.frame1")]['ball_displacement'] <= params['kickoff_ball_max_displacement']).all():
                    pattern.append(['kickoff', last_five_seconds.iloc[0]['player_id'],
                                (last_five_seconds.iloc[0]['x'],last_five_seconds.iloc[0]['y']),
                                last_five_seconds.iloc[0]['ball_z'],'0'])
                else:
                    not_kickoff = next_frame.loc[next_frame['distance'] < params['pz_setpiece']]
                    if not not_kickoff.empty:
                        pattern.append(['free kick', not_kickoff.iloc[0]['player_id'],
                                        (not_kickoff.iloc[0]['x'], not_kickoff.iloc[0]['y']),
                                        next_frame.iloc[0]['ball_z'], '0'])
                    else:
                        pattern.append(['undefined', '',(np.nan, np.nan), 0,'0'])
            else:
                if row.penalty != False:  
                    player_ids = row.penalty['player_id'].split(';')
                    if len(player_ids) > 1:
                        msg = f'WARNING, MORE THAN ONE PLAYER AT PENALTY SPOT, check {row.time1}'
                        if verbose:
                            print(msg)
                        log_incidence[row.frame1] = msg
                        logging.warning(f"{segment_id} - {msg}")
                        
                    penalty_players = next_frame.loc[(next_frame['player_id'].isin(player_ids)) &
                                                  (next_frame['distance'] < params['pz_setpiece'])]
                    if not penalty_players.empty:      
                        pattern.append(['penalty kick', penalty_players.iloc[0]['player_id'],
                                      (penalty_players.iloc[0]['x'],penalty_players.iloc[0]['y']),
                                      next_frame.iloc[0]['ball_z'], '0']) 
                    else:
                        not_penalty = next_frame.loc[next_frame['distance'] < params['pz_setpiece']]
                        if not not_penalty.empty:
                            pattern.append(['free kick', not_penalty.iloc[0]['player_id'],
                                        (not_penalty.iloc[0]['x'],not_penalty.iloc[0]['y']),
                                        next_frame.iloc[0]['ball_z'], '0'])
                        else:
                            pattern.append(['undefined', '',(np.nan, np.nan), 0, '0'])
                
                else:
                    if row.corner != False:
                        pattern.append(check_cornerkick(row, next_frame, params))
                    else:
                        if row.throwin != False:
                            pattern.append(check_throwin(row, next_frame, params))
                        else:
                            ## In this check we investigate all players to take a goal kick, not only goalies
                            pattern.append(check_goalkick(next_frame, params))    

    pattern = pd.DataFrame(pattern, columns=['event','player_id','player_pos', 'ball_z','step'])
    dead_ball_intervals = pd.concat([dead_ball_intervals, pattern], axis=1)

    ## For all the undefined set pieces from complete triggers, do another loop and use incomplete triggers, only if 
    ## triggering player is absent at first in-play frame (useful for HEI data where players were not tracked if off pitch)
    for row in dead_ball_intervals.itertuples():
        if row.event != 'undefined':
            continue
        next_frame = td[pd.eval("td.frame == row.frame1+1")].sort_values(by='distance', ascending=True)
        next_frame_players = next_frame['player_id'].tolist()
        
        dead_ball_intervals.at[row.Index,'step'] = '1'
        if row.corner_incomplete != False and row.corner_incomplete['player_id'].split(';')[0] not in next_frame_players:
            dead_ball_intervals.at[row.Index,'event'] = 'corner kick'
            dead_ball_intervals.at[row.Index,'player_id'] = row.corner_incomplete['player_id'].split(';')[0]
            dead_ball_intervals.at[row.Index,'player_pos'] = row.corner_incomplete['player_pos'][0]
            
        elif row.throwin_incomplete != False and row.throwin_incomplete['player_id'].split(';')[0] not in next_frame_players:
            dead_ball_intervals.at[row.Index,'event'] = 'throw-in'
            dead_ball_intervals.at[row.Index,'player_id'] = row.throwin_incomplete['player_id'].split(';')[0]
            dead_ball_intervals.at[row.Index,'player_pos'] = row.throwin_incomplete['player_pos'][0]
        else:
            
            dead_ball_intervals.at[row.Index,'step'] = '2'
            if row.kickoff != False:
                dead_ball_intervals.at[row.Index,'event'] = 'kickoff'
                dead_ball_intervals.at[row.Index,'player_id'] = row.kickoff['player_id'].split(';')[0]
                dead_ball_intervals.at[row.Index,'player_pos'] = row.kickoff['player_pos'][0]

            elif row.penalty != False:
                dead_ball_intervals.at[row.Index,'event'] = 'penalty kick'
                dead_ball_intervals.at[row.Index,'player_id'] = row.penalty['player_id'].split(';')[0]
                dead_ball_intervals.at[row.Index,'player_pos'] = row.penalty['player_pos'][0]

            elif row.corner != False:
                dead_ball_intervals.at[row.Index,'event'] = 'corner kick'
                dead_ball_intervals.at[row.Index,'player_id'] = row.corner['player_id'].split(';')[0]
                dead_ball_intervals.at[row.Index,'player_pos'] = row.corner['player_pos'][0]

            elif row.throwin != False:
                dead_ball_intervals.at[row.Index,'event'] = 'throw-in'
                dead_ball_intervals.at[row.Index,'player_id'] = row.throwin['player_id'].split(';')[0]
                dead_ball_intervals.at[row.Index,'player_pos'] = row.throwin['player_pos'][0]

            elif row.goalkick != False:
                dead_ball_intervals.at[row.Index,'event'] = 'goal kick'
                dead_ball_intervals.at[row.Index,'player_id'] = row.goalkick['player_id']
                dead_ball_intervals.at[row.Index,'player_pos'] = row.goalkick['player_pos'][0]

            else:         
                dead_ball_intervals.at[row.Index,'event'] = 'free kick'
                dead_ball_intervals.at[row.Index,'player_id'] = next_frame.iloc[0]['player_id']
                dead_ball_intervals.at[row.Index,'player_pos'] = (next_frame.iloc[0]['x'],next_frame.iloc[0]['y'])            

    ## Finally, we store in-game kickoff and penalty kick triggers as they will be useful when determining set pieces
    ## in the presence of tracking data errors
    kickoff_ingame_dict = {}
    ingame_kickoff = kickoff_trigger.query("ball_status != 'dead'").set_index('frame')
    for row in ingame_kickoff.itertuples():
        tm_df = ingame_kickoff.loc[row.Index,'player_id']
        if type(tm_df) == str:
            kickoff_ingame_dict[row.Index] = [tm_df]
        else:
            kickoff_ingame_dict[row.Index] = tm_df.tolist()
            
    
    dead_ball_output = dict()
    dead_ball_output['db_intervals'] = dead_ball_intervals
    dead_ball_output['kickoff_ingame'] = kickoff_ingame_dict
    dead_ball_output['penalty_ingame'] = penalty_trigger.query("ball_status != 'dead'").set_index('frame')['player_id'].to_dict()


    return dead_ball_output, log_incidence


def set_dead_ball_events(model, db_intervals, ball_data, inplay_triggers, params, log_incidence):
    
    """
    This routine establishes the dead ball events and corresponding set pieces using information from the possession model, player configuration
    when the ball is dead via db_intervals and also inplay_triggers for cases when the boolean is inaccurate. Specific parts of the routine are 
    commented for better explainability.

    :param model: Dataframe of events
    :param db_intervals: Dataframe of dead ball intervals, with start/end time and potential set piece
    :param ball_data: Dataframe of ball data
    :param inplay_triggers: Dictionary of dataframes for kickoff and penalty triggers while the ball is in-play
    :param params: Dictionary containing all the relevant parameters
    :param log_incidence: Dictionary where the incidences when running the possession model are logged

    :return: Events dataframe with 'dead_ball_event' and 'from_set_piece' columns added, updated log_incidence and db_intervals. The latter
    is returned because we collapse dead ball intervals whenever the boolean switched to in-play but the game remained dead.
    """ 

    
    control_frames = model.query("event_name != 'no control'")['frame'].tolist()
    total_frames = model.iloc[0]['total_frames']
    model.insert(8,'from_set_piece','')
    model.insert(8,'dead_ball_event','')
    model.set_index('frame', inplace=True)
    
    frame0 = db_intervals['frame0'].tolist()
    frame1 = db_intervals['frame1'].tolist()
    tracking_game_id = params['game_id']
    half = model.iloc[0]['half']
    corner_spots = [(params['pitch_length']/2,params['pitch_width']/2),
                (params['pitch_length']/2,-params['pitch_width']/2)]
    home_id = params['home_id']
    away_id = params['away_id']
    players_df = params['players_df']

    new_dead_ball = []
    f_new_dead_ball = []
    skip_dead_ball_interval = False
    db_intervals['skipped'] = False
    db_intervals['hyperparams'] = None

    for row in db_intervals.itertuples():
        if skip_dead_ball_interval:
            db_intervals.at[row.Index,'skipped'] = True
            skip_dead_ball_interval = False
            continue

        if row.event == 'end of half':
            ## Goals at the end of the half that is not succeeded by a kickoff will be identified here. We rely on whether the ball crossed
            ## the goal after the last possession loss before the game finished.

            end_of_half_event = False
            if db_intervals.at[row.Index-1,'event'] == 'penalty kick':
                ## Special case if last set piece was a penalty kick

                f_shot = model.loc[:row.frame0,'from_set_piece'].str.contains("penalty")[::-1].idxmax()                
                interval = ball_data.loc[f_shot+1:,['ball_x','ball_y']]
                interval['goal_scored_2d'] = interval.apply(lambda r: check_ball_in_goal_2d((r['ball_x'],r['ball_y']), params), axis=1)                

                ## Check if ball crossed the line
                f_ball_in_2d = interval['goal_scored_2d'].idxmax() if interval['goal_scored_2d'].any() else None  

                ## If ball crossed the line and as not above the crossbar, it can be a goal         
                if f_ball_in_2d is not None and check_ball_in_goal_z(ball_data.at[f_ball_in_2d,'ball_z'], params):
                    ## Shot goes on target
                    model.at[f_shot,'event_name'] = 'shot on target'
                    model.at[f_shot,'step_change'] += '-8'                    
                    if not model.loc[f_shot+1:,'event_name'].str.contains("|".join(['pass','possession'])).any(): 
                        ## There is no pass or possession after the this frame, hence there is a goal
                        f_out = (model.loc[:row.frame0,'event_name']!= "")[::-1].idxmax()
                        msg = f"New goal from penalty recorded at {ball_data.at[f_out, 'game_time']}"
                        log_incidence[f_out] = msg
                        if params['verbose']:
                            print(msg)
                        model.at[f_shot,'event_outcome'] = 'goal'
                        model.at[f_out,'dead_ball_event'] = 'goal'                        
                        model.at[f_out, 'step_change'] += '-5'
                        end_of_half_event = True
                    else:
                        ## Another player becomes in possession after shot, hence it's not goal
                        model.at[f_shot,'event_outcome'] = 'unsuccessful'                                                 
                else:
                    ## Shot does not go on target
                    model.at[f_shot,'event_name'] = 'shot not on target' 
                    model.at[f_shot,'event_outcome'] = 'unsuccessful' 
                    model.at[f_shot,'step_change'] += '-8'
                    end_of_half_event = True
                    
            else:
                frame1_last_interval = db_intervals.at[row.Index-1,'frame1']
                interval = ball_data.loc[frame1_last_interval+1:,['ball_x','ball_y']]
                interval['goal_scored_2d'] = interval.apply(lambda r: check_ball_in_goal_2d((r['ball_x'],r['ball_y']), params), axis=1)
                
                ## Check if ball crossed the line
                f_ball_in_2d = interval['goal_scored_2d'].idxmax() if interval['goal_scored_2d'].any() else None
                ## If ball crossed the line and as not above the crossbar, it can be a goal
                if f_ball_in_2d is not None and check_ball_in_goal_z(ball_data.at[f_ball_in_2d,'ball_z'], params):
                    ## Check the last frame where the ball had crossed the line
                    f_ball_in = (interval['goal_scored_2d'] == True)[::-1].idxmax()
                    if not model.loc[f_ball_in:,'event_name'].str.contains("|".join(['pass','possession'])).any(): 
                        ## There is no pass or possession after the this frame, hence there is a goal
                        f_out = (model.loc[frame1_last_interval:f_ball_in,'event_name']!= "")[::-1].idxmax()
                        model.at[f_out,'dead_ball_event'] = 'goal'
                        model.at[f_out, 'step_change'] += '-5'
                        msg = f"New goal recorded at {model.at[f_out, 'event_time']}"
                        log_incidence[f_out] = msg
                        if params['verbose']:
                            print(msg)
                            
                        f_shot = (model.loc[frame1_last_interval:f_ball_in,'event_name'].str.contains('pass'))[::-1].idxmax()
                        
                        model.at[f_shot,'event_name'] = 'shot on target'
                        model.at[f_shot,'event_outcome'] = 'goal'
                        model.at[f_shot,'step_change'] += '-8'
                        end_of_half_event = True
            
            if not end_of_half_event:
                try:
                    ## Establish `end of half` on the last frame where a player was in control
                    f_out = (model.loc[db_intervals.at[row.Index-1,'frame1']:row.frame0,'event_name']!= "")[::-1].idxmax()
                    model.at[f_out,'dead_ball_event'] = 'end of half'
                except:
                    pass
            continue          
              

        ## If the next pass occurs after the following dead ball interval and during the current interval between play resuming and stopping
        ## the ball has not moved substantially (both in cumulative and end-start terms), we collapse the current dead ball interval with 
        ## the following one and do not assign a set piece at the end of current interval, as it is most likely an error in the boolean     

        next_control_frames = model.loc[row.frame1:,'event_name'] != ''
        skip_dead_ball_interval = False
        if next_control_frames.empty:
            ## If there is no other frame in the game with a player in control, we are at the end of the game
            last_event = True
        else:
            last_event = False
            f_0 = next_control_frames.idxmax()

            f_pass = model.loc[f_0:,'event_name'].str.contains("pass").idxmax()
            if row.Index > 0 and row.Index < db_intervals.shape[0]-1 and f_pass > db_intervals.at[row.Index+1,'frame0']:

                ## We compute the distance covered by the ball between the end of the current interval and the start of the following interval
                next_frame0 = db_intervals.at[row.Index+1,'frame0']
                this_ball_traj = ball_data.loc[row.frame1+1:next_frame0-1].copy()
                this_ball_traj['in_pitch'] = this_ball_traj[['ball_x','ball_y']].apply(lambda x: in_pitch_ball(x, params), axis=1)
                distances_to_check = (np.sqrt(this_ball_traj.ball_x.diff()**2 + this_ball_traj.ball_y.diff()**2).sum(),
                                      np.sqrt((this_ball_traj.iloc[-1].ball_x - this_ball_traj.iloc[0].ball_x)**2 + (this_ball_traj.iloc[-1].ball_y - this_ball_traj.iloc[0].ball_y)**2))
                
                ## We make an exception if both this segment and the next one resume in throw-ins and the ball has entered and left the pitch
                ## during the interval between play resuming and stopping (consecutive throw-ins close by)
                both_throw_ins = (this_ball_traj['in_pitch'] == 1).any() and row.event == 'throw-in' and db_intervals.at[row.Index+1,'event'] == 'throw-in'

                ## If the ball has moved and there were no consecutive throw-ins, we skip this dead ball interval
                ball_not_moved = distances_to_check[0] < params['dead_interval_ball_max_displacement'] and distances_to_check[1] < params['dead_interval_ball_max_displacement']  
                if ball_not_moved and not both_throw_ins:
                    skip_dead_ball_interval = True

                db_intervals.at[row.Index+1,'hyperparams'] = distances_to_check


        this_row = {c:getattr(row,c) for c in db_intervals.columns}

        ## We clean-up the end of the current interval according to the following interval
        if skip_dead_ball_interval:
            this_player_id = db_intervals.at[row.Index+1,'player_id']
            this_time1 = db_intervals.at[row.Index+1,'time1']
            this_event = db_intervals.at[row.Index+1,'event']
            this_frame1 = db_intervals.at[row.Index+1,'frame1']
            this_player_pos = db_intervals.at[row.Index+1,'player_pos']
            next_control_frames = model.loc[this_frame1:,'event_name'] != ''
            if next_control_frames.empty:
                last_event = True
            else:
                last_event = False
                f_0 = next_control_frames.idxmax()

            this_row['player_id'] = this_player_id
            this_row['time1'] = this_time1
            this_row['event'] = this_event
            this_row['frame1'] = this_frame1
            this_row['player_pos'] = this_frame1
            msg = f"Segment ending at {row.time1} prolongated to {this_time1} in {params['segment_id']}"
            model.loc[row.frame1:this_frame1,'event_name'] = 'no control'
            model.loc[row.frame1:this_frame1,'step_change'] += '-19'

            log_incidence[this_frame1] = msg
            if params['verbose']:
                print(msg)
        else:
            this_player_id = row.player_id
            this_time1 = row.time1
            this_event = row.event
            this_frame1 = row.frame1
            this_player_pos = row.player_pos


        ## If no data is present when play restarts, we do not assign any set piece
        if this_event == 'no data':
            continue

        ## If this is the last dead ball interval of the game or the player that we found is the executor is not in possession on the next
        ## control frame, we impute the set piece event as it corresponds to the boolean still being dead when the game resumes
        if last_event or (this_player_id not in model.at[f_0, 'player_id'].split(';')):
            
            f_pass = this_frame1
            try:            
                model.loc[f_pass] = {'event_time':this_time1, 'half':half, 'ball_status':'in_play',
                                        'event_name':'pass','dead_ball_event':'','from_set_piece':this_event,
                                        'player_id':this_player_id, 
                                         'team_id':players_df.at[this_player_id,'tracking_team_id'],                                        
                                        'ball_pos':(ball_data.at[this_frame1,'ball_x'],
                                                    ball_data.at[this_frame1,'ball_y'],
                                                    ball_data.at[this_frame1,'ball_z']),
                                         'ball_dir_1':ball_data.at[this_frame1, 'direction_1'], 
                                         'xy':str(this_player_pos),
                                        'step_change':'a','is_goalkeeper':True if this_player_id in params['goalkeeper_ids'] else False,
                                         'ball_changed_direction':True, 'ball_changed_speed':True,
                                        'loss_gain':'', 'event_outcome':'',
                                        'contest':''}
            except:
                if 'kickoff' in this_event:
                    msg = f"{this_event} at {this_time1} is imputed to {this_player_id}"
                    log_incidence[this_frame1] = msg
                    if params['verbose']:
                        print(msg)
                    model.at[f_pass,'from_set_piece'] = this_event
                    model.at[f_pass,'event_name'] = 'pass'
                    model.at[f_pass,'player_id'] = this_player_id
                    model.at[f_pass,'xy'] = str(this_player_pos)
                    model.at[f_pass,'team_id'] = players_df.at[this_player_id,'tracking_team_id']
                    model.at[f_pass,'step_change'] += '-e'
                    model.at[f_pass,'contest'] = ''

                else:
                    msg = f"{this_event} at {this_time1} cannot be imputed to {this_player_id} since players in CZ are {model.at[f_0, 'player_id']} and ball is not dead"
                    log_incidence[this_frame1] = msg
                    if params['verbose']:
                        print(msg)
                continue

            model.sort_values(by='frame', ascending=True, inplace=True)
            dbe = params['db_from_sp'][this_event]
            
        else:

            ## Establish info for the first frame with player in control
            if model.at[f_0,'event_name'] == 'no control':
                model.at[f_0,'event_name'] = 'pass'
                model.at[f_0,'step_change'] = '7'
            elif 'reception' in model.at[f_0,'event_name']:
                model.at[f_0,'event_name'] = 'possession'

            f_pass = model.loc[f_0:,'event_name'].str.contains("pass").idxmax()

            ## if first pass is made not by first player in control, enforce pass made by first player in control
            if this_player_id not in model.at[f_pass,'player_id'].split(';'):
                f_pass = model.loc[f_0:f_pass,'player_id'].str.contains(this_player_id)[::-1].idxmax()
                model.at[f_pass,'event_name'] = 'pass'
                            
            if this_event == 'free kick':
                ## If the set piece is a free kick, we first make sure it is not something else and free kick has been assigned by elimination                
                this_player_traj = model.loc[f_0:f_pass].copy()
                this_player_traj[['x_p','y_p']] = this_player_traj.apply(lambda p: get_displacement_player_frames(p, this_player_id),
                                                               axis=1, result_type='expand')
                
                ## Evaluate the distance traveled by the player and ball from the first fame with player in control to the pass
                player_has_moved = np.sqrt(this_player_traj.x_p.diff()**2 + this_player_traj.y_p.diff()**2).sum() > params.get('set_piece_max_displacement',0)
                this_ball_traj = ball_data.loc[f_0:f_pass]
                ball_has_moved = np.sqrt(this_ball_traj.ball_x.diff()**2 + this_ball_traj.ball_y.diff()**2).sum() > params.get('set_piece_max_displacement',0)
                player_and_ball_have_moved = player_has_moved and ball_has_moved

                if (player_and_ball_have_moved or this_player_traj['x_p'].isnull().any()): 
                    ## If player/ball have displaced before inbounding the ball or the set piece player ceases to appear in the interval 
                    ## between f0 and fpass, we then check for a potential other set piece using the information on db_intervals.
                    ## Most likely the boolean switched to in-play when the set piece had already been taken, hence the motion of players and ball
                    new_event, dbe, frame, log_incidence = new_sp_event(this_row, params, ball_data, log_incidence)                    
                    if dbe != 'foul':
                        new_event['is_goalkeeper'] = True if new_event['player_id'] in params['goalkeeper_ids'] else False                                                        
                        model.loc[frame] = new_event 
                        model.sort_values(by='frame', ascending=True, inplace=True)  
                        f_pass = frame
                    else:
                        dbe = 'foul?'
                        model.at[f_pass, 'from_set_piece'] = 'free kick?'
                        model.at[f_pass, 'step_change'] += '-6' 
     
                else:
                    ## If the above conditions are not satisfied, we still check for other set pieces before assigning free kick. 
                    ## This is of importance when the ball is actually dead but boolean is in-play, hence we use the triggers
                    ## while the ball is (wrongly) in-play
                    pass_player = model.loc[f_pass]
                    xy_player = eval(pass_player['xy'].split(';')[0])
                    team_player = pass_player['team_id'].split(';')[0]
                    
                    new_event_added = False
                    if pass_player['player_id'].split(';')[0] in inplay_triggers['kickoff'].get(f_pass, []):
                        set_piece = 'kickoff'
                        step_change = 'd'
                    elif pass_player['player_id'].split(';')[0] == inplay_triggers['penalty'].get(f_pass, ''):
                        set_piece = 'penalty kick'
                        step_change = 'd'
                    elif corner_trigger(xy_player, np.sign(params['active_goal'][team_player][0]), corner_spots) <= params.get('tol_ck',0):
                        set_piece = 'corner kick'
                        step_change = 'd'
                    elif throwin_trigger(xy_player, params) <= 0:
                        set_piece = 'throw-in'
                        step_change = 'd'
                    elif pass_player['player_id'].split(';')[0] in params['goalkeeper_ids'] and goalkick_trigger(xy_player, np.sign(params['active_goal'][team_player][0]), params):
                        set_piece = 'goal kick'
                        step_change = 'd'
                    else:
                        set_piece = 'free kick'
                        step_change = '-6'
                    
                    if not new_event_added:
                        dbe = params['db_from_sp'][set_piece]
                        model.at[f_pass, 'from_set_piece'] = set_piece
                        model.at[f_pass, 'step_change'] += step_change
                        model.at[f_pass, 'event_name'] = 'pass'
            else:
                ## If the set piece is not a free kick, we assign the set piece as it is
                dbe = params['db_from_sp'][this_event]
                model.at[f_pass, 'from_set_piece'] = this_event
                model.at[f_pass, 'step_change'] += '-6'
        
        ## Minor modification to ensure the first event after a set piece pass is not `possession` but `reception`. 
        next_reception = [cf for cf in control_frames if cf > f_pass]
        if len(next_reception) > 0 and model.at[next_reception[0],'event_name'] == 'possession':
            model.at[next_reception[0],'event_name'] = 'reception'
            model.at[next_reception[0],'step_change'] += '-3'
            

        ## Below we assign the dead ball event prior to the set piece. The first part accounts for edge cases where there was no
        ## in-game event prior to the dead ball interval, hence the dead ball interval is revoked. The second part accounts for the 
        ## special treatment of goals
        if row.Index > 0:
            if (model.loc[:row.frame0,'event_name']!= "").empty:
                msg = f"No prior event for frame {row.frame0}, event {this_event} at {model.at[f_pass,'event_time']} revoked"
                log_incidence[row.frame0] = msg
                if params['verbose']:
                    print(msg)
                model.at[f_pass,'event_name'] = model.at[f_pass,'loss_gain']
                model.at[f_pass, 'from_set_piece'] = ''
                model.at[f_pass, 'step_change'] += '-r'
                continue
            else:
                f_out = (model.loc[:row.frame0,'event_name']!= "")[::-1].idxmax()
                if model.at[f_out,'from_set_piece'] == 'kickoff start':
                    msg = f"No prior event for frame {row.frame0}, event {this_event} at {model.at[f_pass,'event_time']} revoked"
                    log_incidence[row.frame0] = msg
                    if params['verbose']:
                        print(msg)
                    model.at[f_pass,'event_name'] = model.at[f_pass,'loss_gain']
                    model.at[f_pass, 'from_set_piece'] = ''
                    model.at[f_pass, 'step_change'] += '-r'
                    continue
            
            model.at[f_out,'dead_ball_event'] = dbe
            model.at[f_out, 'step_change'] += '-5'

            
            ## Special treatment if there is a goal
            if dbe == 'goal':
                ## Need to state shot on target, which is either last pass or 
                ## needs to be implemented if within that in-play interval there were no passes
                defending_team = players_df.at[this_player_id,'tracking_team_id']
                if defending_team == home_id:
                    attacking_team = away_id 
                else:
                    attacking_team = home_id
                    
                ## The frame where the goal is scored is the last one with a pass
                f_goal = (model.loc[:row.frame0,'event_name'].str.contains('pass'))[::-1].idxmax()
                                
                ## Own goal if at shot time there is only one player from the team that will then kick off.
                ## Data often gets messy when multiple players are involved in a goal, therefore we introduce two corrections
                ## 1) If in the previous control frame there was a loss with an opponent (contest or possession), we assume 
                ## the shot came there and override own goal
                ## 2) We look for possession losses in the frames between f_goal and the ball going dead (f_out) to ensure 
                ## there were shots after f_goal
       
                shot_by_kickoff_team = len(set(model.at[f_goal,'team_id'].split(';'))) == 1 and model.at[f_goal,'team_id'].split(';')[0] == defending_team
                interval = model.loc[f_goal+1:f_out]
                f_after_shot = np.nan
                loss_after_shot = False
                if interval['loss_gain'].str.contains('loss').any():
                    for rr in interval.itertuples():
                        if 'loss' in rr.loss_gain and rr.event_name != 'no control':
                            f_after_shot = rr.Index
                            loss_after_shot = True
                            loss_player_id = rr.player_id.split(';')[0]
                            loss_team_id = rr.team_id.split(';')[0]
                            shot_by_kickoff_team = len(set(rr.team_id.split(';'))) == 1 and rr.team_id.split(';')[0] == defending_team

                if shot_by_kickoff_team:

                    try:
                        opponent_framebefore_owngoal = 'loss' in model.at[f_goal-1,'loss_gain'] and attacking_team in model.at[f_goal-1,'team_id'].split(';')
                        f_framebefore = f_goal-1
                    except:
                        opponent_framebefore_owngoal = False
                        f_framebefore = np.nan
                        
                    if loss_after_shot:
                        model.at[f_after_shot,'event_name'] = 'shot on target'
                        model.at[f_after_shot,'step_change'] += '-11e'
                        model.at[f_after_shot,'event_outcome'] = 'goal'
                        
                    elif not loss_after_shot and opponent_framebefore_owngoal:
                        model.at[f_framebefore,'event_name'] = 'shot on target'
                        model.at[f_framebefore,'step_change'] += '-11f'
                        model.at[f_framebefore,'event_outcome'] = 'goal'
                        model.at[f_goal,'event_name'] = 'no control'
                        model.at[f_goal,'step_change'] += '-4'
                    else:
                        model.at[f_goal,'event_name'] = 'own goal'
                        model.at[f_goal,'event_outcome'] = 'goal'
                        
                        
                else:
                    if loss_after_shot:
                        model.at[f_after_shot,'event_name'] = 'shot on target'
                        model.at[f_after_shot,'step_change'] += '-11g'
                        model.at[f_after_shot,'event_outcome'] = 'goal'
                    else:
                        model.at[f_goal,'event_name'] = 'shot on target'
                        model.at[f_goal,'step_change'] += '-11h'
                        model.at[f_goal,'event_outcome'] = 'goal'

 
       
    return model.reset_index().sort_values(by='frame',ascending=True).reset_index(drop=True), log_incidence, db_intervals



def new_sp_event(row, params, ball_data, log_incidence):
    """
    This routine creates a new set piece event with the time, frame, player_id and set piece nature specified in row. It also annotates it 
    in the log_incidence dictionary.

    :param row: Information on a dead ball interval, including start/end time, set piece that follows and player to execute it. It comes from the db_intervals dataframe
    :param params: Dictionary containing all the relevant parameters
    :param ball_data: Dataframe of ball data
    :param log_incidence: Dictionary where the incidences when running the possession model are logged
    :return: Dictionary with metadata for new set piece event
    """ 

    players_df = params['players_df']   
    frame = row['frame1']
    new_event = {'event_time':row['time1'], 'half':row['time1'].get_half(), 'ball_status':'in_play',
                                    'event_name':'pass','dead_ball_event':'',
                                    'step_change':'b',                 
                                     'ball_changed_direction':True, 'ball_changed_speed':True,
                                    'loss_gain':'', 'event_outcome':'','contest':''}   
    if row['kickoff'] != False:   
        new_event['from_set_piece'] = 'kickoff'
        new_event['player_id'] = row['kickoff']['player_id'].split(';')[0]
        player_pos = row['kickoff']['player_pos'][0]
        new_event['team_id'] = players_df.at[new_event['player_id'],'tracking_team_id']
        dbe = params['db_from_sp']['kickoff']
        msg = f"New kickoff at {row['time1']}"
        log_incidence[row['frame1']] = msg
        if params['verbose']:
            print(msg)
        
    elif row['penalty'] != False:   
        new_event['from_set_piece'] = 'penalty kick'
        new_event['player_id'] = row['penalty']['player_id'].split(';')[0]
        player_pos = row['penalty']['player_pos'][0]
        new_event['team_id'] = players_df.at[new_event['player_id'],'tracking_team_id']
        dbe = params['db_from_sp']['penalty kick']
        msg = f"New penalty at {row['time1']}"
        log_incidence[row['frame1']] = msg
        if params['verbose']:
            print(msg)
            
    elif row['corner'] != False:   
        new_event['from_set_piece'] = 'corner kick'
        new_event['player_id'] = row['corner']['player_id'].split(';')[0]
        player_pos = row['corner']['player_pos'][0]
        new_event['team_id'] = players_df.at[new_event['player_id'],'tracking_team_id']
        dbe = params['db_from_sp']['corner kick']
        
    elif row['throwin'] != False:   
        new_event['from_set_piece'] = 'throw-in'
        new_event['player_id'] = row['throwin']['player_id'].split(';')[0]
        player_pos = row['throwin']['player_pos'][0]
        new_event['team_id'] = players_df.at[new_event['player_id'],'tracking_team_id']
        dbe = params['db_from_sp']['throw-in']
        
    elif row['corner_incomplete'] != False:   
        new_event['from_set_piece'] = 'corner kick'
        new_event['player_id'] = row['corner_incomplete']['player_id'].split(';')[0]
        player_pos = row['corner_incomplete']['player_pos'][0]
        new_event['team_id'] = players_df.at[new_event['player_id'],'tracking_team_id']
        frame = row['corner_incomplete']['frame']
        dbe = params['db_from_sp']['corner kick']
        
    elif row['throwin_incomplete'] != False:   
        new_event['from_set_piece'] = 'throw-in'
        new_event['player_id'] = row['throwin_incomplete']['player_id'].split(';')[0]
        player_pos = row['throwin_incomplete']['player_pos'][0]
        new_event['team_id'] = players_df.at[new_event['player_id'],'tracking_team_id']
        frame = row['throwin_incomplete']['frame']
        dbe = params['db_from_sp']['throw-in'] 
        
    elif row['goalkick'] != False:   
        new_event['from_set_piece'] = 'goal kick'
        new_event['player_id'] = row.goalkick['player_id'].split(';')[0]
        player_pos = row.goalkick['player_pos']
        new_event['team_id'] = players_df.at[new_event['player_id'],'tracking_team_id']
        dbe = params['db_from_sp']['throw-in']  
        
    else:
        player_pos = (np.nan, np.nan)
        dbe = 'foul' 
        
    new_event['xy'] = str((player_pos[0], player_pos[1]))
    new_event['event_time'] = ball_data.at[frame,'game_time']    
    new_event['ball_pos'] = (ball_data.at[row['frame1'], 'ball_x'],
                             ball_data.at[row['frame1'], 'ball_y'],
                             ball_data.at[row['frame1'], 'ball_z'])
    new_event['ball_dir_1'] = ball_data.at[row['frame1'], 'direction_1']
    
    return new_event, dbe, frame, log_incidence

    


###################################################################################################
########################################### UTILITY FUNCTIONS #####################################
###################################################################################################

def get_displacement_player_frames(row, player_id):
    """
    This routine returns the x-y coordinates of the player ID, as row.player_id may contain multiple player IDs separated by ';'

    :param row: Row of events dataframe
    :param player_id: Player ID to get coordinates from

    :return: List of x-y coordinates for the given frame and player ID
    """ 

    players = row.player_id.split(';')
    if player_id in players:
        return list(eval(row.xy.split(';')[players.index(player_id)]))
    else:
        return [np.nan, np.nan]


def check_ball_in_goal_z(ball_z, params):
    """
    This routine returns True if the z-coordinate of the ball is lower than the height of the goal

    :param ball_z: Current height of the ball
    :return: True if the ball is lower than the height of the goal
    """ 

    if np.isnan(ball_z):
        return True
    
    if ball_z < params['pitch_measures']['goal_height']:
        return True
    else:
        return False


def check_ball_in_goal_2d(ball_pos, params):
    """
    This routine returns True if the ball has entered the goal in the 2D space

    :param ball_pos: Current position of the ball
    :param params: Dictionary containing all the relevant parameters
    :return: True if the ball is inside either the left or right goal in 2D
    """ 

    pitch_measures = params['pitch_measures']
    
    right_goal = ball_pos[0] > params['pitch_length']/2 and ball_pos[0] < params['pitch_length']/2 + pitch_measures['goal_height'] and abs(ball_pos[1]) < pitch_measures['goal_dim']/2
    left_goal = ball_pos[0] < -params['pitch_length']/2 and ball_pos[0] > -params['pitch_length']/2 - pitch_measures['goal_height'] and abs(ball_pos[1]) < pitch_measures['goal_dim']/2
    return right_goal or left_goal



def in_hull(p, other_w_gk, index_wo_gk, active_goal_x):
    """
    This routine returns the location of player with coordinates p in relation to the convex hull of the opposing team

    :param p: Coordinates of player
    :param other_w_gk: x-y coordinates of all players in the opposing team
    :param index_wo_gk: Index of all infielders in the opposing team, because GK is omitted for convex hull calculation
    :param active_goal_x: Direction of attacker for player with coordinates p
    :return: A string containing the location of the player with coordinates p relative to the other team, from the list
    ['in front',behind','flank','inside with X rivals in front','outside with X rivals in front']
    """ 
    
    hull = other_w_gk.iloc[index_wo_gk][['x','y']].to_numpy()
    xy_w_gk = other_w_gk[['x','y']].to_numpy()
    
    if not isinstance(hull,Delaunay):
        hull_res = Delaunay(hull)

    is_inside = hull_res.find_simplex(p)>=0
    
    if is_inside:
        if active_goal_x > 0:
            location = f'inside with {(xy_w_gk[:,0] >= p[0]).sum()} rivals in front'
        else:
            location = f'inside with {(xy_w_gk[:,0] <= p[0]).sum()} rivals in front'
    else:
        if active_goal_x > 0:
            if p[0] <= hull[:,0].min():
                location = 'in front'
            elif p[0] >= hull[:,0].max():
                location = 'behind'
            elif p[1] >= hull[:,1].max() or p[1] <= hull[:,1].min():
                location = 'flank'
            else:                
                location = f'outside with {(xy_w_gk[:,0] >= p[0]).sum()} rivals in front'
        elif active_goal_x < 0:
            if p[0] >= hull[:,0].max():
                location = 'in front'
            elif p[0] <= hull[:,0].min():
                location = 'behind'
            elif p[1] >= hull[:,1].max() or p[1] <= hull[:,1].min():
                location = 'flank'
            else:
                location = f'outside with {(xy_w_gk[:,0] <= p[0]).sum()} rivals in front'
        
    return location


def compute_angle(pos0, pos1, active_goal_x):
    """
    This routine returns the angle in degrees between 0-360 for an event starting in pos0, ending in pos1 with the direction 
    of attack specified by active_goal_x

    :param pos0: Tuple of coordinates from origin of event
    :param pos1: Tuple of coordinates from end of event
    :param active_goal_x: x-coordinate of goal where player is attacking towards
    :return: The value of the angle
    """ 

    if pos1[0] == pos0[0]:
        return 0
    
    if active_goal_x < 0:
        pos0 = (-pos0[0],-pos0[1])
        pos1 = (-pos1[0],-pos1[1])
    ang = 180/np.pi*np.arctan2((pos0[1]-pos1[1]),(pos0[0]-pos1[0]))
    if ang < 0:
        return round(360+ang,4)
    else:
        return round(ang,4)



def in_penalty_box(x, y, left_right_box, params, tol=0):
    """
    This routine evaluates whether the player located in x,y is located in the left/right penalty box, with a buffer tolerance

    :param x: x-coordinate of player
    :param y: y-coordinate of player
    :param left_right_box: -1/+1 to indicate whether the left/right penalty box is checked
    :param params: Dictionary containing all the relevant parameters
    :param tol: Optional parameter to account for a buffer in the penalty box requirement
    :return: True if player is in active penalty box
    """ 

    pitch_measures = params['pitch_measures']
    d = pitch_measures['penbox_len']
    r = params['pitch_length']/2 - d/2
    return np.logical_and(abs(y) <= pitch_measures['penbox_wid']/2 + tol, abs(x - left_right_box*r) <= d/2 + tol)


def in_cross_box(x, y, left_right_box, cross_sign, params):
    """
    This routine evaluates whether the player located in x,y is located in the left/right receiving cross box, 
    which together with other conditions will determine if the previous pass is labeled as a cross or not

    :param x: x-coordinate of player
    :param y: y-coordinate of player
    :param left_right_box: -1/+1 to indicate the direction of attack
    :param cross_sign: positive if the cross comes from the upper half of the pitch and negative otherwise
    :param params: Dictionary containing all the relevant parameters
    :return: True if player is in receiving cross box
    """ 

    pitch_measures = params['pitch_measures']    
    d = pitch_measures['penspot_dim'] + pitch_measures['arc_dim']
    r = params['pitch_length']/2 - d/2
    return np.logical_and(cross_sign*y - pitch_measures['smbx_dim'] - pitch_measures['goal_dim']/2 <= 0, 
                         abs(x - left_right_box*r) <= d/2)


def shot_on_target(ball_x, ball_y, ball_dir, active_goal_x, params):
    """
    This routine evaluates whether the ball located at (ball_x, ball_y) with direction ball_dir is: 
    a) going on target (with a threshold); and
    b) going forward, namely in the direction of attack

    :param ball_x: x-coordinate of ball
    :param ball_y: y-coordinate of ball
    :param ball_dir: Direction of ball
    :param active_goal_x: x-coordinate of goal where player that shot the ball is attacking towards
    :param params: Dictionary containing all the relevant parameters
    :return: Booleans on_target_direction and shot_forward
    """ 

    if ball_dir[0] == 0:
        on_target_direction = False
        shot_forward = False
    else:
        m_dir = ball_dir[1]/ball_dir[0]
        if active_goal_x > 0:
            shot_forward = True if ball_dir[0] > 0 else False
            on_target_direction = abs(m_dir*(params['pitch_length']/2 - ball_x) + ball_y) < params['pitch_measures']['goal_dim']/2 + params.get('on_target_threshold',0) and shot_forward       
        else:
            shot_forward = True if ball_dir[0] < 0 else False
            on_target_direction = abs(m_dir*(-params['pitch_length']/2 - ball_x) + ball_y) < params['pitch_measures']['goal_dim']/2 + params.get('on_target_threshold',0) and shot_forward

    return on_target_direction, shot_forward



def jersey_team(player_id, params):
    """
    This routine returns the `jersey_number (team name)` of a player/s based on the players_df dataframe. If player is not found we return the same variable player_id

    :param player_id: Player_id to assign jersey & team name
    :param params: Dictionary containing all the relevant parameters
    :return: Jersey number (team name)
    """ 

    if type(player_id) == str and player_id != '' and set(player_id.split(';')).issubset(set(params['players_df'].index.tolist())):
        out = []
        for y in player_id.split(';'):            
            z = f"{params['players_df'].at[y,'jersey_number']} ({params['players_df'].at[y,'team_name']})"
            out.append(str(z))
        return ';'.join(out)
    else:
        return player_id

def names(player_id, params):
    """
    This routine returns the name of a player/s based on the players_df dataframe. If player is not found we return the same variable player_id

    :param player_id: Player_id to assign name
    :param params: Dictionary containing all the relevant parameters
    :return: Name of player
    """ 

    if type(player_id) == str and player_id != '' and set(player_id.split(';')).issubset(set(params['players_df'].index.tolist())):
        out = []
        for y in player_id.split(';'):            
            z = f"{params['players_df'].at[y,'name']} ({params['players_df'].at[y,'team_acronym']})"
            out.append(str(z))
        return ';'.join(out)
    else:
        return player_id


def roles(player_id, params):
    """
    This routine returns the role of a player/s based on the players_df dataframe. If player is not found we return the same variable player_id

    :param player_id: Player_id to assign role
    :param params: Dictionary containing all the relevant parameters
    :return: Role of player
    """ 

    if type(player_id) == str and player_id != '' and set(player_id.split(';')).issubset(set(params['players_df'].index.tolist())):
        out = []
        for y in player_id.split(';'):
            z = params['players_df'].at[y,'position']
            out.append(str(z))
        return ";".join(out)
    else:
        return player_id


def in_pitch_player(xy, params):
    """
    This routine returns True if any of the players' coordinates in xy are inside the pitch (with buffers defined in params['player_in_pitch']) or False 
    if all the players are outside the pitch.

    :param xy: Positions of players
    :param params: Dictionary containing all the relevant parameters
    :return: True or False
    """ 

    in_pitch = []
    for player_str in xy.split(';'):
        player = eval(player_str)
        out_sideline = abs(player[1]) >= params['pitch_width']/2 + params['player_in_pitch']['sideline']
        out_endline = abs(player[0]) >= params['pitch_length']/2 + params['player_in_pitch']['endline'] and abs(player[1]) > params['pitch_measures']['goal_dim']/2
        out_ingoal = abs(player[0]) >= params['pitch_length']/2 + params['player_in_pitch']['goal'] and abs(player[1]) <= params['pitch_measures']['goal_dim']/2
        if out_endline or out_sideline or out_ingoal:
            in_pitch.append(False)
        else:
            in_pitch.append(True)
    return any(in_pitch)


def in_pitch_ball(pos, params): 
    """
    This routine returns 0 if the ball is out of the pitch (with a buffer params['ball_in_pitch']) and 1 if the ball is inside the pitch,

    :param pos: Position of ball
    :param params: Dictionary containing all the relevant parameters
    :return: 0 or 1
    """ 

    out_sideline = abs(pos.ball_y) >= params['pitch_width']/2 - params['ball_in_pitch']
    out_endline = abs(pos.ball_x) >= params['pitch_length']/2 - params['ball_in_pitch']
    if out_endline or out_sideline:
        return 0
    else:
        return 1

def pitch_zone(pos, active_goal_x, params):
    """
    This routine returns which zone of the pitch is the player located in based on the definitions of cross zone and shot zone.

    :param pos: Position of player
    :param active_goal_x: x-coordinate of goal where player is attacking towards
    :param params: Dictionary containing all the relevant parameters
    :return: 'other', 'cross' or 'shot'
    """ 
    arc_dim = params['pitch_measures']['arc_dim']
    smbx_dim = params['pitch_measures']['smbx_dim']
    penbox_len = params['pitch_measures']['penbox_len']
    penbox_wid = params['pitch_measures']['penbox_wid']
    penspot_dim = params['pitch_measures']['penspot_dim']
    goal_dim = params['pitch_measures']['goal_dim']

    cross_penarea_triangle = penbox_wid/2-goal_dim/2-smbx_dim
    if active_goal_x > 0:
        penarea_corner = params['pitch_length']/2-penbox_len
        y_cross_low = pos[1] < -penbox_wid/2 + (cross_penarea_triangle/penbox_len)*(pos[0] - penarea_corner)
        y_cross_up  = pos[1] > penbox_wid/2 - (cross_penarea_triangle/penbox_len)*(pos[0] - penarea_corner)

        if pos[0] >= arc_dim and ((abs(pos[1]) > penbox_wid/2) or (y_cross_low and pos[0] >= penarea_corner) or (y_cross_up and pos[0] >= penarea_corner)):
            return 'cross'
        elif pos[0] < arc_dim:
            return 'other'
        else:
            return 'shot'
    else:
        penarea_corner = -params['pitch_length']/2+penbox_len
        y_cross_low = pos[1] < -penbox_wid/2 - (cross_penarea_triangle/penbox_len)*(pos[0] - penarea_corner)
        y_cross_up  = pos[1] >  penbox_wid/2 + (cross_penarea_triangle/penbox_len)*(pos[0] - penarea_corner)
        if pos[0] <= -arc_dim and ((abs(pos[1]) > penbox_wid/2) or (y_cross_low and pos[0] <= penarea_corner) or (y_cross_up and pos[0] <= penarea_corner)):
            return 'cross'
        elif pos[0] > -arc_dim:
            return 'other'
        else:
            return 'shot'  


def match_player_interval(this_string, ref_player):
    """
    Returns True if the string this_string contains the player ref_player

    :param this_string: String containing one or more players separated by ';'
    :param ref_player: Player/s to look for in this_string
    :return: True if this_string contains any of the players in ref_player
    """ 

    if len(this_string.split(';')) == 1 and len(ref_player.split(';')) == 1:
        return this_string == ref_player
    else:
        return not set(this_string.split(';')).isdisjoint(ref_player.split(';'))


def unroll_tuple(this_tuple, dim):
    """
    Extracts the dimension dim of a concatenation of tuples separated by ';'

    :param this_tuple: Concatenation of tuples
    :param dim: Dimension to extract from each tuple
    :return: A string with the dimension dim of each tuple separated by ';'
    """ 

    return ';'.join([str(eval(s)[dim]) for s in this_tuple.split(';')]) if this_tuple != '' else ''   


def ball_direction(pos0, pos1):
    """
    Computes the direction of the ball

    :param pos0: Initial position
    :param pos1: End position
    :return: Direction of ball as a tuple
    """ 

    dx_ = pos1[0]-pos0[0]
    dy_ = pos1[1]-pos0[1]
    if (dx_ == 0) & (dy_==0):
        return (0,0)
    else:
        return (dx_/np.sqrt(dx_**2+dy_**2), dy_/np.sqrt(dx_**2+dy_**2))


def count_player_in_penalty_box(params):

    """
    This routine computes how many infielders are in their respective attacking penalty box at every frame
    :param params: Dictionary containing all the relevant parameters        
    :return: Dictionary the players of each team that are in the opposing penalty box at every frame, to determine crosses    
    """ 

    players_in_areas = {}
    td = params['td']
    home_id = params['home_id']
    away_id = params['away_id']

    all_players = td[pd.eval("~td.player_id.isin(params['goalkeeper_ids'])")][['x','y','team_id','active_goal','frame']].copy()
    home_td = all_players[pd.eval("all_players.team_id == home_id")][['x','y','frame','active_goal']]
    home_td['in_penbox'] = in_penalty_box(home_td['x'].values, home_td['y'].values,
                                                  home_td['active_goal'].values, params, tol=params['in_area'])
    players_in_areas[f"{away_id}_penbox"] = home_td[['frame','in_penbox']].groupby('frame',
                                                         sort=False).agg(sum).rename(columns={'in_penbox':'count'})

    away_td = all_players[pd.eval("all_players.team_id == away_id")][['x','y','frame','active_goal']]
    away_td['in_penbox'] = in_penalty_box(away_td['x'].values, away_td['y'].values,
                                                  away_td['active_goal'].values, params, tol=params['in_area'])
    players_in_areas[f"{home_id}_penbox"] = away_td[['frame','in_penbox']].groupby('frame',
                                                         sort=False).agg(sum).rename(columns={'in_penbox':'count'})

    return players_in_areas


def in_cone_vision(p0, v0, p, ang):

    """
    This routine examines if the players in locations p are within the cone of vision of player p0 with orientation v0. A player p
    is within the cone of vision of they are within +- ang of the vector v0.

    :param p0: x-y coordinates of reference player
    :param v0: Orientation of reference player
    :param p: Numpy array containing the x-y coordinates of other players, one per row
    :param ang: Angle in radians that specifies the cone of vision
    :return: Boolean numpy array where for each player returns True if that player is within cone of vision of reference player
    """ 

    if v0[0] == 0 and v0[1] == 0:
        return np.asarray(p.shape[0]*[True])
    else:

        ## Calculate the vectors between the other players and the ball handler
        v = p - p0

        ## Calculate the angle between v0 and each vector
        
        num = np.sum(v0 * v, axis=1)
        den = (np.linalg.norm(v0) * np.linalg.norm(v, axis=1))
        angle = [np.arccos(np.divide(num[i],d)) if d != 0 else 0 for i,d in enumerate(den)]

        ## Check if the angle is within the specified range
        return np.array([a <= ang*np.pi/180 for a in angle])



def events_to_json(model):

    """
    This routine converts the events dataframe into a list of dictionaries, one entry per event, and dumps it to a JSON

    :param model: Events dataframe
    :return: JSON containing a list of dictionaries, one per row in the events dataframe
    """ 

    events_list = []
    string_cols = ['ball_control','event_name','event_outcome','from_set_piece','dead_ball_event',
                        'contest','description',
                        'assist_event_id','previous_loss_event_id','next_gain_event_id','next_loss_event_id',
                        'player_location','ball_from','collected_at','team_possession','pressure','defensive_mode'] 
    tuple_cols = ['xy']
    player_cols = ['jersey','team_name','player_id','team_id','player_role']
    num_cols = ['distance_traveled_recipient','incoming_angle','outgoing_angle',
                'packing_pass','packing_possession']

    pc_cols = ['pc_gain','pc_passing_line','within_vision']
    for row in model.itertuples():
        this_event = dict()
        this_event['frame'] = row.frame
        this_event['event_id'] = row.event_id
        this_event['event_time'] = str(row.event_time)
        this_event['time_ms'] = row.time_ms
        this_event['half'] = row.half
        this_event['ball_position'] = tuple(row.ball_pos)
        this_event['ball_speed_mps'] = row.ball_speed
        this_event['home_score'] = row.home_score
        this_event['away_score'] = row.away_score
        

        for col in string_cols:
            if hasattr(row,col) and getattr(row,col) != '':
                this_event[col] = getattr(row,col).split(';') if ';' in getattr(row,col) else getattr(row,col)

        if row.contest == '':
            for col in player_cols:
                if hasattr(row,col) and getattr(row,col) != '':
                    this_event[col] = getattr(row,col).split(';')

            for col in tuple_cols:
                if hasattr(row,col) and getattr(row,col) != '':
                    this_event[col] = [eval(x) for x in getattr(row,col).split(';')]
        else:
            for col_ in player_cols:
                col = 'contest_'+col_
                if hasattr(row,col) and getattr(row,col) != '':
                    this_event[col] = getattr(row,col).split(';')

            for col_ in tuple_cols:
                col = 'contest_'+col_
                if hasattr(row,col) and getattr(row,col) != '':
                    this_event[col] = [eval(x) for x in getattr(row,col).split(';')]

        for col in num_cols:
            if hasattr(row,col) and not np.isnan(getattr(row,col)):
                this_event[col] = getattr(row,col)

        for col in pc_cols:
            if hasattr(row,col) and getattr(row,col) is not None:
                this_event[col] = getattr(row,col)

        if hasattr(row,'distance_traveled_ball_x') and not np.isnan(getattr(row,'distance_traveled_ball_x')):
            this_event['distance_traveled_ball'] = (row.distance_traveled_ball_x,row.distance_traveled_ball_y)
        if hasattr(row,'home_span_x') and not np.isnan(getattr(row,'home_span_x')):
            this_event['home_span'] = (row.home_span_x,row.home_span_y)
            this_event['away_span'] = (row.away_span_x,row.away_span_y)

        events_list.append(this_event)

    return json.dumps(events_list)


