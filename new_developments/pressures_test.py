import pandas as pd
import numpy as np

def create_team_pressure_v2(events, team_possession, params):

    """
    This routine collapses individual pressures into team pressures that encompass frames where no player has possession as well 
    (as long as they are shorter than `max_team_press_cutoff`), and merges them in the `team_possession` dataframe. 
    It also creates a `team_pressures` dataframe with start/end time of each team pressure   

    :param model: Events dataframe
    :param team_possession: Team possession dataframe
    :param params: Dictionary containing all the relevant parameters    
    :return: Team possession dataframe updated, team pressures dataframe
    """   
    
    pressures = events.query("pressure != ''").copy().reset_index(drop=True)
    team_possession['pressure'] = ''
    team_possession.set_index('time_ms', inplace=True)
    ball_data = params['ball_data'].set_index('time_ms')
        
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
        ## Potential change #0 (active)
        no_interruptions = (team_possession.loc[end_time:press.time_ms,'is_home'].diff().dropna() == 0).all()  
        
        ## Potential change #3 (active)
        no_high_ball = (ball_data.loc[end_time:press.time_ms,'ball_z'] > params['pressure']['high_ball_z']).sum() < params['fps']*params['pressure']['high_ball_duration']
        # no_high_ball = True        
        
        if this_team == press.team_possession and press.time_ms - end_time < params['pressure']['max_team_press_cutoff']*1000 and no_oop and no_high_ball and no_interruptions:
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
    return team_possession.reset_index(), team_pressures
        



def detect_pressure_v2(model, params):

    """
    This routine generates pressures. The pressures are generated for each particular frame, and then they are transformed to team pressures
    using the create_team_pressures routine.      

    :param model: Events dataframe
    :param team_possession: Team possession dataframe
    :param params: Dictionary containing all the relevant parameters    
    :return: Events and team possession dataframes updated, team pressures dataframe
    """     

    gk_threshold = params['pressure']['gk_z_threshold']
    infielder_threshold = params['pressure']['infielder_z_threshold']
    
    ## Potential change #5 (not active)
    # possession = model.query("ball_control == 'possession' and contest == '' and from_set_piece == '' and team_id == team_possession and ((ball_z < @gk_threshold and is_goalkeeper == 1) or (ball_z < @infielder_threshold and is_goalkeeper == 0))").copy()
    possession = model.query("ball_control == 'possession' and contest == '' and from_set_piece == '' and team_id == team_possession").copy()
    ## Potential change #1 (active)
    contest = model.query("contest in ['contest','ground duel'] and from_set_piece == '' and ball_z < @infielder_threshold").copy()
    possession = pd.concat([possession, contest], ignore_index=True).sort_values(by='frame',ascending=True)
    possession.set_index('frame', drop=False,inplace=True)
    model.set_index('event_id', inplace=True)
    
    possession['pressure'] = ''
    possession['pressure_distance'] = ''

    pressure_params = params['pressure']
    td = params['td']
    ball_data = td.drop_duplicates(subset='frame')[['time_ms','frame','ball_status']].set_index('frame').copy()
    
    ## Create tracking data group by frame for faster access
    first_frame = td.iloc[0]['frame']
    new_frames = list(possession['frame'])
    old_frames = [max(f-int(pressure_params['delay_secs']*params['fps']), first_frame) for f in new_frames]
    all_frames = new_frames + old_frames
    home_grouped_now = dict(tuple(td[pd.eval("td.frame in new_frames and td.is_home == 1 and ((td.distance < pressure_params['radius'] and td.speed>pressure_params['min_speed']) or td.distance < pressure_params['radius_back'])")][['frame','player_id','x','y']].groupby('frame')))
    away_grouped_now = dict(tuple(td[pd.eval("td.frame in new_frames and td.is_home == 0 and ((td.distance < pressure_params['radius'] and td.speed>pressure_params['min_speed']) or td.distance < pressure_params['radius_back'])")][['frame','player_id','x','y']].groupby('frame')))
    all_grouped_past = dict(tuple(td[pd.eval("td.frame in old_frames")][['frame','x','y','player_id']].groupby('frame')))
    all_grouped_now = dict(tuple(td[pd.eval("td.frame in new_frames")][['frame','speed','player_id','vx','vy']].groupby('frame')))
    
    ball_z_all = td.drop_duplicates(subset=['frame']).set_index('frame')['ball_z']
    final_third = False
    team_possession = None
    frame = None
    for current_event in possession.itertuples():
        
        ## Potential change #4 (not active)
#         if current_event.event_name in ['cross','shot on target','shot not on target']:
#             final_third = True
#             team_possession = current_event.team_possession
#             frame = current_event.frame
#             continue
        
#         if final_third:
#             if (ball_data.loc[frame:current_event.frame,'ball_status'] == 'dead').any():
#                 final_third = False    
#             else:
#                 if current_event.team_possession != team_possession:
#                     final_third = False     
#                 elif current_event.is_gain == True and model.at[current_event.next_loss_event_id,'event_name'] == 'pass':
#                     final_third = False

            
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

        ## Special treatment for goalkeeper being pressed. This can be commented if we restrict the frames with the z threshold above
        if current_player_id in params['goalkeeper_ids']: 

            future_poss = possession.loc[current_event.frame:current_event.frame+pressure_params['lookahead_secs']*params['fps']].iloc[-1]
            if future_poss.player_id.split(';')[0] == current_player_id:
                ball_z_next_sec = ball_z_all.loc[current_event.frame:current_event.frame+pressure_params['lookahead_secs']*params['fps']].mean()
            else:
                ball_z_next_sec = current_event.ball_z
            height_condition = ball_z_next_sec > pressure_params['gk_z_limit'][0] and ball_z_next_sec <= pressure_params['gk_z_limit'][1]
        goalie_hands = current_player_id in params['goalkeeper_ids'] and height_condition and in_active_penalty_box(current_event.ball_x, current_event.ball_y, -active_goal, params, tol=params['in_area'])
        
        
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
                ## In every other case, we examine the opponents that have come closer to the ball handler in the delay_secs period
                last_poss_frame = max(current_event.frame-int(pressure_params['delay_secs']*params['fps']), first_frame)

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
                    
                    ## iterate through opponents
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
                            
                        ## Pull player position for distance check
                        distance_p2p_past = np.sqrt((all_players_past[presser_id]['x']-attacker_past_xy[0])**2+(all_players_past[presser_id]['y'] - attacker_past_xy[1])**2)

                        ## Check if presser is moving closer to the attacker, is within sight/presence of attacker and is moving towards the attacker
                        if distance_p2p < distance_p2p_past and within_radius and towards_attacker:
                            
                            continuous.append(presser_id)
                            distances.append(round(distance_p2p,2))
      
        if len(continuous) > 0:
            possession.at[current_event.Index,'pressure'] = ';'.join(list(set(continuous)))
            possession.at[current_event.Index,'pressure_distance'] = ';'.join([str(d) for d in distances])
        
    possession.reset_index(inplace=True, drop=True)
    model.reset_index(inplace=True)
    
    continuous_pressure = possession.set_index('frame')['pressure'].to_dict()
    model['pressure'] = model['frame'].apply(lambda x: continuous_pressure.get(x,''))
    pressure_distance = possession.set_index('frame')['pressure_distance'].to_dict()
    model['pressure_distance'] = model['frame'].apply(lambda x: pressure_distance.get(x,''))

    return model