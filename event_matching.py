import numpy as np
import pandas as pd
from time import time
from game_time import GameTime


def match_player_interval(this_string,ref_player):
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


def jersey_team(player_id, params, ids='tracking'):
    """
    This routine returns the `jersey_number (team name)` of a player/s based on the players_df dataframe. If player is not found we return the same variable player_id

    :param player_id: Player_id to assign jersey & team name
    :param params: Dictionary containing all the relevant parameters
    :return: Jersey number (team name)
    """ 

    if ids == 'event':
        players_df = params['players_df'].reset_index().set_index('events_player_id')
    else:
        players_df = params['players_df']

    if type(player_id) == str and player_id != '' and set(player_id.split(';')).issubset(set(params['players_df'].index.tolist())):
        out = [] 
        for y in player_id.split(';'):            
            z = f"{params['players_df'].at[y,'jersey_number']} ({params['players_df'].at[y,'team_name']})"
            out.append(str(z))
        return ';'.join(out)
    else:
        return player_id

def map_events_to_tracking(events_player_id, players_df, tracking_name):
    """
    This routine returns the value of the column specified by tracking_name for the player whose ID is given by events_player_id

    :param player_events_id: Player ID on the events data
    :param players_df: Dataframe with player information, indexed by events_player_id
    :param tracking_name: Column to retrieve from players_df
    :return: tracking_name column for the events_player_id of interest
    """ 
    if type(events_player_id) == str and events_player_id != '' and events_player_id in players_df.index:       
        return players_df.at[events_player_id, tracking_name]
    else:
        return events_player_id


        
def event_metrics(confusion_matrix, ratios=None):    
    """
    This routine computes the precision, recall and F1-score metrics for a given confusion matrix. If ratios are supplied, they will be
    used to compute these metrics

    :param confusion_matrix: Confusion matrix
    :param ratios: Tuple of numpy arrays of weights
    :return: Dictionary with aggregate and individual values for precision, recall, f1-score and accuracy
    """ 

    if confusion_matrix.sum() == 0:
        accuracy = np.nan
        precision = (len(confusion_matrix)-1)*[np.nan]
        recall = (len(confusion_matrix)-1)*[np.nan]
    else:
        accuracy = 100*confusion_matrix.trace()/confusion_matrix.sum()
        precision = []
        recall = []
        for i in range(len(confusion_matrix[0,:-1])):
            if confusion_matrix[:,i].sum() > 0:
                precision.append(confusion_matrix[i,i]/confusion_matrix[:,i].sum()*100)
            else:
                precision.append(np.nan)
            if confusion_matrix[i,:].sum() > 0:
                recall.append(confusion_matrix[i,i]/confusion_matrix[i,:].sum()*100)
            else:
                recall.append(np.nan)
    
    f1_score = np.array([0 if p == 0 and recall[i] == 0 else 2*p*recall[i]/(p+recall[i]) for i,p in enumerate(precision)])  
    precision = np.array(precision)
    recall = np.array(recall)

    
    if ratios is None:
        ratios_pre = len(precision)*[1/len(precision)]
        ratios_rec = len(recall)*[1/len(recall)]
    else:
        ratios_pre = ratios[0]
        ratios_rec = ratios[1]
    
    metrics = dict()
    if np.isnan(precision).all():
        avg_precision = np.nan
        w_precision = np.nan
    else:
        avg_precision = np.nanmean(precision)
        w_precision = np.nansum(precision*ratios_pre)  

    if np.isnan(recall).all():
        avg_recall = np.nan
        w_recall = np.nan
    else:
        avg_recall = np.nanmean(recall)
        w_recall = np.nansum(recall*ratios_rec)
    
    if np.isnan(f1_score).all():
        avg_f1_score = np.nan  
        w_f1_score = np.nan 
    else:
        avg_f1_score = np.nanmean(f1_score)
        w_pre = precision*ratios_pre
        w_rec = precision*ratios_pre
        w_f1_score = 0 if w_precision == 0 and w_recall == 0 else 2*w_precision*w_recall/(w_precision+w_recall)
    

    metrics['accuracy'] = round(accuracy,2)
    metrics['macro_precision'] = round(avg_precision,2)
    metrics['macro_recall'] = round(avg_recall,2)
    metrics['macro_f1_score'] = round(avg_f1_score,2)
    metrics['weighted_precision'] = round(w_precision,2)
    metrics['weighted_recall'] = round(w_recall,2)
    metrics['weighted_f1_score'] = round(w_f1_score,2)
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1_score
    metrics['ratios'] = (ratios_pre, ratios_rec)

    return metrics



def get_official_events_sts(this_game, matching_params):
    """
    This routine reshapes STS events from a dataframe to a dictionary for ease of matching

    :param this_game: Game structure containing auto-eventing parameters and official STS events
    :param matching_params: Dictionary containing all the relevant parameters for matching
    :return: Dictionary with STS passing, saving and dueling events, as well as all events and tags used for passes
    """ 
    params = this_game['params']
    half = int(params['segment_id'].split('_')[-1])
    players_df = params['players_df'].set_index('events_player_id')
    include_oba = matching_params['include_oba']

    all_ev = this_game['events_sts'][['event_id', 'time', 'event_name', 'event_outcome',
                                    'player_1','player_2','team_1', 'team_2', 
                                    'game_section','from_set_piece', 'type']].copy()
    all_ev.dropna(subset=['player_1'], inplace=True)

    all_ev.insert(0,'event_time','')
    all_ev['event_time'] = all_ev['time'].apply(lambda x: GameTime.from_event_time(half, x))

    all_ev['team_1'] = all_ev['player_1'].apply(map_events_to_tracking, args=(players_df, 'tracking_team_id'))
    all_ev['team_2'] = all_ev['player_2'].apply(map_events_to_tracking, args=(players_df, 'tracking_team_id'))
    all_ev['player_1'] = all_ev['player_1'].apply(map_events_to_tracking, args=(players_df, 'tracking_player_id'))
    all_ev['player_2'] = all_ev['player_2'].apply(map_events_to_tracking, args=(players_df, 'tracking_player_id'))

    all_ev.insert(5,'jersey_team_1','')
    all_ev['jersey_team_1'] = all_ev['player_1'].apply(jersey_team, args=(params,))
    all_ev.insert(6,'jersey_team_2','')
    all_ev['jersey_team_2'] = all_ev['player_2'].apply(jersey_team, args=(params,))

    
    
    all_ev.rename(columns={'time':'time_ms'}, inplace=True)
    
    list_of_passes = ['shot','pass','cross','fair play','goal','clearance','own goal']

    if include_oba:
        passes_ev = all_ev.loc[all_ev['event_name'].isin(list_of_passes +['other_ball_action'])].copy()
    else:
        passes_ev = all_ev.loc[all_ev['event_name'].isin(list_of_passes)].copy()


    all_ev = all_ev.sort_values(by='time_ms', ascending=True).reset_index(drop=True)                              
    passes_ev = passes_ev.drop(columns=['player_2','team_2']).reset_index(drop=True)


    all_ev['game_section'].fillna('', inplace=True)
    passes_ev['game_section'].fillna('', inplace=True)                                
    all_ev['from_set_piece'].fillna('open play', inplace=True)
    passes_ev['from_set_piece'].fillna('open play', inplace=True)
    saves_ev = pd.DataFrame()

    return {'passes':passes_ev, 'all':all_ev, 'saves':saves_ev,'pass_tags': list_of_passes}#, 'set_piece':setpiece_ev}


def get_official_events_dcu(this_game, matching_params):
    """
    This routine reshapes DCU events from a dataframe to a dictionary for ease of matching

    :param this_game: Game structure containing auto-eventing parameters and official DCU events
    :param matching_params: Dictionary containing all the relevant parameters for matching
    :return: Dictionary with DCU passing, saving and dueling events, as well as all events and tags used for passes
    """ 

    params = this_game['params']
    half = int(params['segment_id'].split('_')[-1])
    players_df = params['players_df'].set_index('events_player_id')


    all_ev = this_game['events_dcu'][['event_id', 'time', 'event', 'outcome','from_player_id',
                     'to_player_id','team_id', 'event_type', 'action_type', 'from_set_piece']].copy()

    all_ev.dropna(subset=['from_player_id'], inplace=True)
    all_ev.rename(columns={'time':'time_ms',
                            'event':'event_name','outcome':'event_outcome',
                            'from_player_id':'player_1','to_player_id':'player_2',
                            'team_id':'team_1','event_type':'type'}, inplace=True)

    all_ev.insert(0,'event_time','')
    all_ev['event_time'] = all_ev['time_ms'].apply(lambda x: GameTime.from_event_time(half, x))

    all_ev.insert(5,'jersey_team_1','')
    all_ev['jersey_team_1'] = all_ev['player_1'].apply(jersey_team, args=(params,))
    all_ev.insert(6,'jersey_team_2','')
    all_ev['jersey_team_2'] = all_ev['player_2'].apply(jersey_team, args=(params,))


    list_of_passes = ['shot','pass','cross','clearance','own goal']
    passes_ev = all_ev.loc[all_ev['event_name'].isin(list_of_passes)].copy()
    passes_ev = passes_ev.drop(columns=['player_2']).reset_index(drop=True)

    all_ev['action_type'].fillna('', inplace=True)
    all_ev['from_set_piece'].fillna('', inplace=True)
    passes_ev['from_set_piece'].fillna('open play', inplace=True)

    ### exclude save_attempt and no_save_attempt
    saves_ev = all_ev.query("event_name.isin(['goal prevention','aerial control']) and ~action_type.str.contains('attempt')").copy()

    duels_ev = all_ev.query("event_name.isin(['duel','tackle','aerial duel','physical duel','take on against'])").reset_index(drop=True)

    return {'passes':passes_ev, 'all':all_ev, 'saves':saves_ev,'pass_tags': list_of_passes, 'duels':duels_ev}#, 'set_piece':setpiece_ev}



def matching_algorithm_sts(this_game, matching_params):
  """
    This routine matches the auto-events in this_game['models']['events'] to the STS events. 
    Specific parts of the routine are commented for better explainability. 


    :param this_game: Game structure containing auto-events and official STS events
    :param matching_params: Dictionary containing all the relevant parameters for matching
    :return: Dictionary with the performance metrics and with actual matched events for matching passes, set pieces, saves, duels
  """ 

  params = this_game['params']
  model = this_game['models']['events']
  model['frame'] = model['frame'].astype(str)
  official_ev = this_game['events_sts'].copy()
  thres = matching_params['thresh_for_detection']
  include_oba = matching_params['include_oba']
  factor = 1000
    
  passes_ev = official_ev['passes']
  final_model = model.query("dead_ball_event == 'goal' | event_name!=''").copy()
  final_model['from_set_piece'] = final_model['from_set_piece'].replace('','open play')
  final_model.replace({' and retain':'',' and deflect':''}, inplace=True, regex=True)
  passes_model = final_model.query("event_name.str.contains('|'.join(['pass','shot','cross','own']))", engine='python').reset_index(drop=True)
  
  pass_tags = official_ev['pass_tags']
  saves_ev = official_ev['saves']

  ## Matching Set pieces if performed by same team
  dbe_model = passes_model.query("from_set_piece != 'open play'").reset_index(drop=True)
  dbe_off = passes_ev.query("from_set_piece != 'open play'", engine='python').reset_index(drop=True)

  first_filter = 15
  matching_dbe = []
  for row in dbe_off.itertuples():

    t0 = row.time_ms - first_filter*1000
    t1 = row.time_ms + first_filter*1000

    within_minute = dbe_model.loc[(dbe_model['time_ms'] >= t0) &
                                  (dbe_model['time_ms'] <= t1)].copy()
    
    if within_minute.empty:
      continue
    else:
      within_minute['same_player'] = within_minute['player_id'].apply(match_player_interval,args=(row.player_1,))
      within_minute['same_team'] = within_minute['team_id'].apply(match_player_interval,args=(row.team_1,))
      within_minute['time_delta'] = within_minute['event_time'].apply(lambda x: abs(x.to_event_time()[1] - row.time_ms)/1000)
      within_minute = within_minute.loc[within_minute['same_team'] == True]

      if within_minute.empty:
        continue
      else:
        within_minute = within_minute.sort_values(by='time_delta').reset_index()    
        within_minute_matched_dbe = within_minute.loc[0]
        matching_dbe.append([within_minute_matched_dbe['time_delta'], within_minute_matched_dbe['frame'],
                             row.event_id, within_minute_matched_dbe['event_time']])
    del within_minute

  match_dbe = pd.DataFrame(matching_dbe,columns=['time_delta','frame_model','event_id','event_time_model'])

  min_deltas = match_dbe.groupby(['frame_model'])['time_delta'].transform(min)
  match_dbe = match_dbe.loc[match_dbe['time_delta'] == min_deltas].reset_index(drop=True)
  match_dbe['wave'] = 'dbe'

  ## Matching open-play passes/shots/crosses
  passes_ev_open = passes_ev.query("from_set_piece == 'open play'").reset_index(drop=True)
  passes_model_open = passes_model.query("from_set_piece == 'open play'", engine='python').reset_index(drop=True)

  matching_events = []
  for row in passes_ev_open.itertuples():

    t0 = row.time_ms - thres*1000
    t1 = row.time_ms + thres*1000
        
    within_minute = passes_model_open.loc[(passes_model_open['time_ms'] >= t0) &
                                          (passes_model_open['time_ms'] <= t1)].copy()
    if within_minute.empty:
      continue
    else:
      within_minute['same_player'] = within_minute['player_id'].apply(match_player_interval,args=(row.player_1,))
      within_minute['time_delta'] = within_minute['event_time'].apply(lambda x: abs(x.to_event_time()[1] - row.time_ms)/1000)
      within_minute = within_minute.loc[(within_minute['same_player'] == True) & (within_minute['time_delta'] < thres)]

      if within_minute.empty:
        continue
      else:
        within_minute = within_minute.sort_values(by='time_delta').reset_index()    
        within_minute_matched = within_minute.loc[0]
        matching_events.append([within_minute_matched['time_delta'], within_minute_matched['frame'],
                                row.event_id, within_minute_matched['event_time']])
    del within_minute

  first_matching = pd.DataFrame(matching_events,columns=['time_delta','frame_model','event_id','event_time_model'])

  min_deltas = first_matching.groupby(['frame_model'])['time_delta'].transform(min)
  first_matching = first_matching.loc[first_matching['time_delta'] == min_deltas].reset_index(drop=True)
  first_matching['wave'] = 'first'


  unmatched_off = passes_ev_open.loc[~passes_ev_open['event_id'].isin(first_matching['event_id'].unique()),'event_id']
  unmatched_off = passes_ev_open[['event_time','event_name','jersey_team_1','event_id']].merge(unmatched_off, how='right', on='event_id')

  unmatched_model = passes_model_open.loc[~passes_model_open['frame'].isin(first_matching['frame_model'].tolist()),
                                     ['event_time','event_name','jersey_team','frame','player_id','time_ms']]


  ############## TRY TO MATCH UNMATCHED MODEL EVENTS TO ANY OFFICIAL EVENT
  # again, only omit from set piece events, not saves
  second_matching = []

  # if include_oba:
  #   all_ev_second = official_ev['all'].query("from_set_piece == 'open play'")

  #   not_acceptable_tags = ['substitution','red card','yellowred card','yellow card',
  #                            'run','player not sent off','penalty not awarded',
  #                            'chance without shot','foul','ball claimed',
  #                            'offside','sitter prevented offside','caution','goalkeeper change',
  #                         'caution teamofficial','foul','fault execution','period start','period end',
  #                         'start penalty shootout','offside','drop ball','kickoff','corner kick','throw-in',
  #                         'throw-in','free kick','goal kick','penalty kick']              
                             

  #   list_matched_official = first_matching['event_id'].tolist()
    
  #   candidate = all_ev_second.loc[(~all_ev_second['event_id'].isin(list_matched_official)) &
  #                                 (~all_ev_second['player_1'].isnull()) & (~all_ev_second['event_name'].isin(not_acceptable_tags))]

  #   for row in unmatched_model.itertuples():

  #     t0 = row.time_ms - thres*1000
  #     t1 = row.time_ms + thres*1000

  #     within_minute = candidate.loc[(candidate['time_ms'] >= t0) &
  #                                   (candidate['time_ms'] <= t1)].copy()
  #     if within_minute.empty:
  #       continue
  #     else:
  #       for rr in within_minute.itertuples():
  #         if rr.event_name == 'tackle':
  #           within_minute.loc[rr.Index,'same_player'] = match_player_interval(",".join([rr.player_1, rr.player_2]), row.player_id)

  #         else:
  #           within_minute.loc[rr.Index,'same_player'] = match_player_interval(rr.player_1, row.player_id)
          
  #       within_minute['time_delta'] = within_minute['time_ms'].apply(lambda x: abs(x - row.time_ms)/factor)
  #       within_minute = within_minute.loc[(within_minute['same_player'] == True) & (within_minute['time_delta'] < thres)]

  #       if within_minute.empty:
  #         continue
  #       else:
  #         within_minute = within_minute.sort_values(by='time_delta').reset_index()    
  #         within_minute_matched = within_minute.loc[0]
  #         second_matching.append([within_minute_matched['time_delta'], row.frame,
  #                                 within_minute_matched['event_id'], row.event_time])


  #   second_matching = pd.DataFrame(second_matching,columns=['time_delta','frame_model','event_id','event_time_model'])

  #   min_deltas = second_matching.groupby(['event_id'])['time_delta'].transform(min)
  #   second_matching = second_matching.loc[second_matching['time_delta'] == min_deltas].reset_index(drop=True)
  #   second_matching['wave'] = 'second'
  # else:
  second_matching = pd.DataFrame(second_matching,columns=['time_delta','frame_model','event_id','event_time_model'])

  ## Join open-play passes and set pieces
  all_ev = official_ev['all']
  events_match = pd.concat([first_matching, second_matching, match_dbe], ignore_index=True).sort_values(by='frame_model')
  events_match = events_match.merge(all_ev[['event_time','event_name','from_set_piece','jersey_team_1',
                                            'event_id','game_section','event_outcome']], how='left', on='event_id')
  events_match = events_match.merge(passes_model[['frame','event_name','from_set_piece','dead_ball_event','step_change']].rename(columns={'frame':'frame_model'}), how='left', on='frame_model',suffixes=('','_model'))
  
  finally_unmatched_model = passes_model.loc[~passes_model['frame'].isin(events_match['frame_model'].unique()),'frame']
  finally_unmatched_model = passes_model[['event_time','event_name','from_set_piece',
                                        'jersey_team','player_id','frame',
                                        'dead_ball_event','step_change']].merge(finally_unmatched_model, how='right', on='frame')
  finally_unmatched_model['source'] = 'model'
  finally_unmatched_model.rename(columns={'frame':'event_id'}, inplace=True)

  finally_unmatched_off = passes_ev.loc[~passes_ev['event_id'].isin(events_match['event_id'].unique()),'event_id']
  finally_unmatched_off = passes_ev[['event_time','event_name','from_set_piece','jersey_team_1',
                                     'player_1','event_id','game_section','event_outcome']].merge(finally_unmatched_off, how='right', on='event_id')

  finally_unmatched_off['source'] = 'official'
  finally_unmatched_off = finally_unmatched_off.rename(columns={'jersey_team_1':'jersey_team',
                                                                'player_1':'player_id'})

  ## remove start of half kickoff
  events_match = events_match.loc[~events_match['game_section'].str.contains('start')]
  finally_unmatched_off = finally_unmatched_off.loc[~finally_unmatched_off['game_section'].str.contains('start')]
  finally_unmatched_model = finally_unmatched_model.loc[~finally_unmatched_model['from_set_piece'].str.contains('start')]    
  events_match.drop(columns=['game_section'], inplace=True)
  finally_unmatched_off.drop(columns=['game_section'], inplace=True)

  unmatched_events = pd.concat([finally_unmatched_model,finally_unmatched_off],ignore_index=True)
  unmatched_events = unmatched_events.sort_values(by='event_time',ascending=True).reset_index(drop=True)


  ## Matching duels
  duels_ev = all_ev.query("event_name == 'duel'").reset_index(drop=True)
  duels_ev['contest'] = duels_ev['type'] + ' duel'
  duels_model = model.query("contest.str.contains('duel')", engine='python').copy().reset_index().rename(columns={'index':'duel_index'})
  
  duel_filter = 10
  matching_duels = []
  for row in duels_ev.itertuples():

    t0 = row.time_ms - duel_filter*1000
    t1 = row.time_ms + duel_filter*1000

    within_minute = duels_model.loc[(duels_model['time_ms'] >= t0) &
                                        (duels_model['time_ms'] <= t1)].copy()
    
    if within_minute.empty:
      continue
    else:
      within_minute['same_player_1'] = within_minute['contest_player_id'].apply(match_player_interval,args=(row.player_1,))
      if row.player_2 is None:
        within_minute['same_player_2'] = True
      else:
        within_minute['same_player_2'] = within_minute['contest_player_id'].apply(match_player_interval,args=(row.player_2,))
      within_minute['time_delta'] = abs(within_minute['time_ms'] - row.time_ms)/1000
      within_minute = within_minute.loc[(within_minute['same_player_1'] == True) & \
                                        (within_minute['same_player_2'] == True) & \
                                        (within_minute['time_delta'] < thres)]

      if within_minute.empty:
        continue
      else:
        within_minute = within_minute.sort_values(by='time_delta').reset_index()    
        within_minute_matched = within_minute.loc[0]
        matching_duels.append([within_minute_matched['time_delta'], within_minute_matched['frame'],
                                row.event_id, within_minute_matched['event_time']])
    del within_minute
  
  duels_ev['players'] = duels_ev['player_1'] + ';' + duels_ev['player_2']
  duels_ev['teams'] = duels_ev['team_1'] + ';' + duels_ev['team_2']
  duels_ev['jersey_team'] = duels_ev['players'].apply(jersey_team, args=(params,))

  duels_match = pd.DataFrame(matching_duels,columns=['time_delta','frame','event_id','event_time_model'])
  duels_match = duels_match.merge(duels_ev[['event_time','contest',
                                          'jersey_team','event_id']], how='left', on='event_id')
  duels_match = duels_match.merge(duels_model[['frame','contest']], how='left', on='frame',suffixes=('','_model'))
  
  unmatched_duels_model = duels_model.loc[~duels_model['frame'].isin(duels_match['frame'].unique()),'frame']
  unmatched_duels_model = duels_model[['event_time','contest',
                                       'contest_player_id','contest_team_id',
                                       'frame']].merge(unmatched_duels_model, how='right', on='frame')
  unmatched_duels_model['jersey_team'] = unmatched_duels_model['contest_player_id'].apply(jersey_team, args=(params,))  
  unmatched_duels_model['source'] = 'model'
  unmatched_duels_model.drop(columns=['contest_player_id','contest_team_id'], inplace=True)
  unmatched_duels_model.rename(columns={'frame':'event_id'}, inplace=True)

  unmatched_duels_off = duels_ev.loc[~duels_ev['event_id'].isin(duels_match['event_id'].unique()),'event_id']
  unmatched_duels_off = duels_ev[['event_time','contest','jersey_team',
                                   'event_id']].merge(unmatched_duels_off, how='right', on='event_id')

  unmatched_duels_off['source'] = 'official'
  
  unmatched_duels = pd.concat([unmatched_duels_model,unmatched_duels_off],ignore_index=True)
  unmatched_duels = unmatched_duels.sort_values(by='event_time',ascending=True).reset_index(drop=True)
  duels_match.drop(columns=['event_id'],inplace=True)


  ## Matching goals
  goals_ev = all_ev.query("event_outcome == 'goal' or event_name == 'own goal'").reset_index(drop=True)
  goals_model_raw = model.query("dead_ball_event == 'goal'").copy().reset_index().rename(columns={'index':'goal_index'})
  if not goals_model_raw.empty:
    ii = []
    ev = []
    for row in goals_model_raw.itertuples():
      j = model.loc[:row.goal_index,'event_name'].str.contains("|".join(['shot','own','cross']))[::-1].idxmax()
      ii.append(j)
      ev.append('own goal' if model.at[j,'event_name'] == 'own goal' else 'goal')
    goals_model = model.loc[ii].copy().reset_index(drop=True)
    goals_model['dbe'] = ev
    
  else:
    goals_model = goals_model_raw.copy()
    goals_model['dbe'] = goals_model['dead_ball_event']
    
  goal_filter = 15
  matching_goals = []
  for row in goals_ev.itertuples():

    t0 = row.time_ms - goal_filter*1000
    t1 = row.time_ms + goal_filter*1000

    within_minute = goals_model.loc[(goals_model['time_ms'] >= t0) &
                                        (goals_model['time_ms'] <= t1)].copy()
    
    if within_minute.empty:
      continue
    else:
      if len(set(within_minute.iloc[0]['team_id'].split(';'))) == 2:
        within_minute['same_team'] = True
      else:
        if (row.event_name == 'goal' and within_minute.iloc[0]['dbe'] == 'goal') or (row.event_name == 'own goal' and within_minute.iloc[0]['dbe'] == 'own goal'):        
          within_minute['same_team'] = within_minute['team_id'].apply(match_player_interval,args=(row.team_1,))
        else:
          within_minute['same_team'] = ~(within_minute['team_id'].apply(match_player_interval,args=(row.team_1,)))

      within_minute['time_delta'] = within_minute['event_time'].apply(lambda x: abs(x.to_event_time()[1] - row.time_ms)/1000)
      within_minute = within_minute.loc[(within_minute['same_team'] == True) & (within_minute['time_delta'] < goal_filter)]

      if within_minute.empty:
        continue
      else:
        within_minute = within_minute.sort_values(by='time_delta').reset_index()    
        within_minute_matched = within_minute.loc[0]
        matching_goals.append([within_minute_matched['time_delta'], within_minute_matched['frame'],
                                row.event_id, within_minute_matched['event_time']])
    del within_minute

  goals_match = pd.DataFrame(matching_goals,columns=['time_delta','frame','event_id','event_time_model'])
  goals_match = goals_match.merge(all_ev[['event_time','event_name',
                                          'jersey_team_1','event_id']].rename(columns={'jersey_team_1':'jersey_team'}), how='left', on='event_id')
  goals_match = goals_match.merge(goals_model[['frame','event_name','dbe','jersey_team']], how='left', on='frame',suffixes=('','_model'))
  
  unmatched_goals_model = goals_model.loc[~goals_model['frame'].isin(goals_match['frame'].unique()),'frame']
  unmatched_goals_model = goals_model[['event_time','event_name','jersey_team',
                                       'player_id','frame','dbe']].merge(unmatched_goals_model, how='right', on='frame')
  unmatched_goals_model['source'] = 'model'
  unmatched_goals_model.rename(columns={'frame':'event_id'}, inplace=True)

  unmatched_goals_off = goals_ev.loc[~goals_ev['event_id'].isin(goals_match['event_id'].unique()),'event_id']
  unmatched_goals_off = goals_ev[['event_time','event_name','jersey_team_1',
                                     'player_1','event_id']].merge(unmatched_goals_off, how='right', on='event_id')

  unmatched_goals_off['source'] = 'official'
  unmatched_goals_off = unmatched_goals_off.rename(columns={'jersey_team_1':'jersey_team',
                                                                'player_1':'player_id'})
  unmatched_goals = pd.concat([unmatched_goals_model,unmatched_goals_off],ignore_index=True)
  unmatched_goals = unmatched_goals.sort_values(by='event_time',ascending=True).reset_index(drop=True)
  goals_match.drop(columns=['event_id'],inplace=True)

 

  ## There are no saves reported in STS
  
  saves_match = pd.DataFrame() 
  unmatched_saves = pd.DataFrame()

  ## Confusion matrices
  results = dict()
  results['open_play'] = dict()  
  results['pass_shot'] = dict()  
  results['pass_shot_cross'] = dict()    
  results['duel'] = dict()
  results['save'] = dict()
  results['goal'] = dict()
  results['set_piece'] = dict()

  matching = dict()
  matching['passes_match'] = events_match
  matching['passes_unmatch'] = unmatched_events
  matching['saves_match'] = saves_match
  matching['saves_unmatch'] = unmatched_saves
  matching['goals_match'] = goals_match
  matching['goals_unmatch'] = unmatched_goals
  matching['duels_match'] = duels_match
  matching['duels_unmatch'] = unmatched_duels


  ## Open-play Passes/Shots/Crosses
  
  if include_oba:        
    pass_tags += 'other ball action'
    
  pass_names = ['pass/shot/cross','none']
  events_match_binary = events_match.query("event_name.str.contains('|'.join(@pass_tags))", engine='python').reset_index(drop=True)
  pm_binary = np.zeros((len(pass_names),len(pass_names)))
  pm_binary[0,0] = events_match_binary.shape[0]
  pm_binary[0,1] = finally_unmatched_off.shape[0]
  pm_binary[1,0] = finally_unmatched_model.shape[0]
  results['open_play']['names'] = pass_names
  results['open_play']['performance'] = pm_binary     

  shot_tags_matching = ['shot','goal']
  pass_tags_matching = [tag for tag in pass_tags if tag not in ['cross']+shot_tags_matching]

  pm_open = np.zeros((4,4))
  pm_open[0,0] = events_match_binary.query("event_name.isin(@pass_tags_matching) and event_name_model == 'pass'").shape[0]
  pm_open[1,1] = events_match_binary.query("event_name.isin(@shot_tags_matching) and event_name_model.str.contains('shot')").shape[0]  
  pm_open[2,2] = events_match_binary.query("event_name == 'cross' and event_name_model == 'cross'").shape[0] 
  pm_open[0,1] = events_match_binary.query("event_name.isin(@pass_tags_matching) and event_name_model.str.contains('shot')").shape[0]
  pm_open[0,2] = events_match_binary.query("event_name.isin(@pass_tags_matching) and event_name_model == 'cross'").shape[0]
  pm_open[1,0] = events_match_binary.query("event_name.isin(@shot_tags_matching) and event_name_model == 'pass'").shape[0]
  pm_open[1,2] = events_match_binary.query("event_name.isin(@shot_tags_matching) and event_name_model == 'cross'").shape[0]
  pm_open[2,0] = events_match_binary.query("event_name == 'cross' and event_name_model == 'pass'").shape[0]
  pm_open[2,1] = events_match_binary.query("event_name == 'cross' and event_name_model.str.contains('shot')").shape[0]
  pm_open[-1,0] = finally_unmatched_model.query("event_name == 'pass'").shape[0]
  pm_open[-1,1] = finally_unmatched_model.query("event_name.str.contains('shot')").shape[0]
  pm_open[-1,2] = finally_unmatched_model.query("event_name == 'cross'").shape[0]
  pm_open[0,-1] = finally_unmatched_off.query("event_name.isin(@pass_tags_matching)").shape[0]
  pm_open[1,-1] = finally_unmatched_off.query("event_name.isin(@shot_tags_matching)").shape[0]
  pm_open[2,-1] = finally_unmatched_off.query("event_name == 'cross'").shape[0]
  results['pass_shot_cross']['names'] = ['pass','shot','cross','none']
  results['pass_shot_cross']['performance'] = pm_open

  pass_tags_matching = [tag for tag in pass_tags if tag not in shot_tags_matching]
  pm_open = np.zeros((3,3))
  pm_open[0,0] = events_match_binary.query("event_name.isin(@pass_tags_matching) and event_name_model.isin(['pass','cross'])").shape[0]
  pm_open[1,1] = events_match_binary.query("event_name.isin(@shot_tags_matching) and event_name_model.str.contains('shot')").shape[0]  
  pm_open[0,1] = events_match_binary.query("event_name.isin(@pass_tags_matching) and event_name_model.str.contains('shot')").shape[0]
  pm_open[1,0] = events_match_binary.query("event_name.isin(@shot_tags_matching) and event_name_model.isin(['pass','cross'])").shape[0]
  pm_open[-1,0] = finally_unmatched_model.query("event_name.isin(['pass','cross'])").shape[0]
  pm_open[-1,1] = finally_unmatched_model.query("event_name.str.contains('shot')").shape[0]
  pm_open[0,-1] = finally_unmatched_off.query("event_name.isin(@pass_tags_matching)").shape[0]
  pm_open[1,-1] = finally_unmatched_off.query("event_name.isin(@shot_tags_matching)").shape[0]
  results['pass_shot']['names'] = ['pass/cross','shot','none']
  results['pass_shot']['performance'] = pm_open
    
  ## Saves
  save_names = ['save','claim','reception','none']
  pm_save = np.zeros((len(save_names),len(save_names)))
  if not saves_ev.empty:      
      pm_save[0,0] = saves_match.query("event_name == 'save' and event_name_model == 'save'", engine='python').shape[0]
      pm_save[0,1] = saves_match.query("event_name == 'save' and event_name_model == 'claim'", engine='python').shape[0]
      pm_save[0,2] = saves_match.query("event_name == 'save' and event_name_model.str.contains('loose')", engine='python').shape[0]
      pm_save[0,-1] = unmatched_saves_off.query("event_name == 'save'").shape[0]

      pm_save[1,0] = saves_match.query("event_name.isin(['claim','ball claimed']) and event_name_model == 'save'", engine='python').shape[0]
      pm_save[1,1] = saves_match.query("event_name.isin(['claim','ball claimed']) and event_name_model == 'claim'", engine='python').shape[0]
      pm_save[1,2] = saves_match.query("event_name.isin(['claim','ball claimed']) and event_name_model.str.contains('loose')", engine='python').shape[0]
      pm_save[1,-1] = unmatched_saves_off.query("event_name.isin(['claim','ball claimed'])", engine='python').shape[0]

      pm_save[2,0] = saves_match.query("~event_name.isin(['claim','ball claimed','save']) and event_name_model == 'save'", engine='python').shape[0]
      pm_save[2,1] = saves_match.query("~event_name.isin(['claim','ball claimed','save']) and event_name_model == 'claim'", engine='python').shape[0]
      pm_save[2,2] = saves_match.query("~event_name.isin(['claim','ball claimed','save']) and event_name_model.str.contains('loose')", engine='python').shape[0]
      pm_save[2,-1] = unmatched_saves_off.query("~event_name.isin(['claim','ball claimed','save'])", engine='python').shape[0]

      pm_save[-1,0] = unmatched_saves_model.query("event_name == 'save'", engine='python').shape[0]
      pm_save[-1,1] = unmatched_saves_model.query("event_name == 'claim'", engine='python').shape[0]
      pm_save[-1,2] = unmatched_saves_model.query("event_name.str.contains('loose')", engine='python').shape[0]

  results['save']['names'] = save_names
  results['save']['performance'] = pm_save
  
  ## Goals
  goal_names = ['goal','own goal','none']

  pm_goal = np.zeros((len(goal_names),len(goal_names)))
  pm_goal[0,0] = goals_match.query("event_name == 'goal' and dbe == 'goal'", engine='python').shape[0]
  pm_goal[0,1] = goals_match.query("event_name == 'goal' and dbe == 'own goal'", engine='python').shape[0]
  pm_goal[1,0] = goals_match.query("event_name == 'own goal' and dbe == 'goal'", engine='python').shape[0]  
  pm_goal[1,1] = goals_match.query("event_name == 'own goal' and dbe == 'own goal'", engine='python').shape[0]
  pm_goal[0,-1] = unmatched_goals_off.query("event_name != 'own goal'").shape[0]
  pm_goal[1,-1] = unmatched_goals_off.query("event_name == 'own goal'").shape[0]
  pm_goal[-1,0] = unmatched_goals_model.query("dbe == 'goal'").shape[0]
  pm_goal[-1,1] = unmatched_goals_model.query("dbe == 'own goal'").shape[0]

  results['goal']['names'] = goal_names
  results['goal']['performance'] = pm_goal

  ## Duels
  duel_names = ['air duel','ground duel','none']

  pm_duel = np.zeros((len(duel_names),len(duel_names)))
  pm_duel[0,0] = duels_match.query("contest == 'air duel' and contest_model == 'air duel'", engine='python').shape[0]
  pm_duel[0,1] = duels_match.query("contest == 'air duel' and contest_model == 'ground duel'", engine='python').shape[0]
  pm_duel[1,0] = duels_match.query("contest == 'ground duel' and contest_model == 'air duel'", engine='python').shape[0]  
  pm_duel[1,1] = duels_match.query("contest == 'ground duel' and contest_model == 'ground duel'", engine='python').shape[0]
  pm_duel[0,-1] = unmatched_duels_off.query("contest == 'air duel'").shape[0]
  pm_duel[1,-1] = unmatched_duels_off.query("contest == 'ground duel'").shape[0]
  pm_duel[-1,0] = unmatched_duels_model.query("contest == 'air duel'").shape[0]
  pm_duel[-1,1] = unmatched_duels_model.query("contest == 'ground duel'").shape[0]

  results['duel']['names'] = duel_names
  results['duel']['performance'] = pm_duel


  ## Set pieces
  events_match_setpiece = events_match.query("event_name.str.contains('|'.join(@pass_tags))", engine='python').reset_index(drop=True)
  results_origin = dict()
  set_names = ['kickoff','penalty kick','corner kick','throw-in','goal kick','free kick','free kick?','none']
  pm_set = np.zeros((len(set_names),len(set_names)))
  for i, name_i in enumerate(set_names[:-1]):
    for j, name_j in enumerate(set_names[:-1]):
      pm_set[i,j] = events_match_setpiece.loc[(events_match_setpiece['from_set_piece'] == name_i) & (events_match_setpiece['from_set_piece_model'] == name_j)].shape[0]
  
  for j, name_j in enumerate(set_names[:-1]):
    pm_set[-1,j] = finally_unmatched_model.query("from_set_piece == @name_j and event_name.str.contains('|'.join(['pass','shot','cross']))", engine='python').shape[0]
  
  for i, name_i in enumerate(set_names[:-1]):
    pm_set[i,-1] = finally_unmatched_off.query("from_set_piece == @name_i and event_name.str.contains('|'.join(@pass_tags))", engine='python').shape[0]

  results['set_piece']['names'] = set_names
  results['set_piece']['performance'] = pm_set
  results['pass_tags'] = pass_tags


  return results, matching



def matching_algorithm_dcu(this_game, matching_params):

  """
    This routine matches the auto-events in this_game['models']['events'] to the STS events. 
    Specific parts of the routine are commented for better explainability. 


    :param this_game: Game structure containing auto-events and official STS events
    :param matching_params: Dictionary containing all the relevant parameters for matching
    :return: Dictionary with the performance metrics and with actual matched events for matching passes, set pieces, saves, duels
  """ 

  params = this_game['params']
  model = this_game['models']['events']
  model['frame'] = model['frame'].astype(str)
  official_ev = this_game['events_dcu'].copy()
  thres = matching_params['thresh_for_detection']
  factor = 1000
    
  passes_ev = official_ev['passes']
  all_ev = official_ev['all']
  final_model = model.query("dead_ball_event == 'goal' | event_name!=''").copy()
  final_model['from_set_piece'] = final_model['from_set_piece'].replace('','open play')
  final_model.replace({' and retain':'',' and deflect':''}, inplace=True, regex=True)
  passes_model = final_model.query("event_name.str.contains('|'.join(['pass','shot','cross','own']))", engine='python').reset_index(drop=True)
  
  pass_tags = official_ev['pass_tags']
  saves_ev = official_ev['saves']


  ## Matching Set pieces if performed by same team
  dbe_model = passes_model.query("from_set_piece != 'open play'").reset_index(drop=True)
  dbe_ev = all_ev.query("from_set_piece != ''", engine='python').reset_index(drop=True)

  first_filter = 15
  matching_dbe = []
  for row in dbe_ev.itertuples():

    t0 = row.time_ms - first_filter*1000
    t1 = row.time_ms + first_filter*1000

    within_minute = dbe_model.loc[(dbe_model['time_ms'] >= t0) &
                                  (dbe_model['time_ms'] <= t1)].copy()
    
    if within_minute.empty:
      continue
    else:
      within_minute['same_team'] = within_minute['team_id'].apply(match_player_interval,args=(row.team_1,))
      within_minute['time_delta'] = within_minute['event_time'].apply(lambda x: abs(x.to_event_time()[1] - row.time_ms)/1000)

      within_minute = within_minute.loc[within_minute['same_team'] == True]

      if within_minute.empty:
        continue
      else:
        within_minute = within_minute.sort_values(by='time_delta').reset_index()    
        within_minute_matched_dbe = within_minute.loc[0]
        matching_dbe.append([within_minute_matched_dbe['time_delta'], within_minute_matched_dbe['frame'],
                             row.event_id, within_minute_matched_dbe['event_time']])
    del within_minute

  match_dbe = pd.DataFrame(matching_dbe,columns=['time_delta','frame_model','event_id','event_time_model'])

  min_deltas = match_dbe.groupby(['frame_model'])['time_delta'].transform(min)
  match_dbe = match_dbe.loc[match_dbe['time_delta'] == min_deltas].reset_index(drop=True)
  match_dbe['wave'] = 'dbe'

  ## Matching passes/shots/crosses
  passes_ev_open = passes_ev.reset_index(drop=True)
  passes_model_open = passes_model.reset_index(drop=True)

  matching_events = []
  for row in passes_ev_open.itertuples():

    t0 = row.time_ms - thres*1000
    t1 = row.time_ms + thres*1000
        
    within_minute = passes_model_open.loc[(passes_model_open['time_ms'] >= t0) &
                                          (passes_model_open['time_ms'] <= t1)].copy()
    if within_minute.empty:
      continue
    else:
      within_minute['same_player'] = within_minute['player_id'].apply(match_player_interval,args=(row.player_1,))
      within_minute['time_delta'] = within_minute['event_time'].apply(lambda x: abs(x.to_event_time()[1] - row.time_ms)/1000)
      within_minute = within_minute.loc[(within_minute['same_player'] == True) & (within_minute['time_delta'] < thres)]

      if within_minute.empty:
        continue
      else:
        within_minute = within_minute.sort_values(by='time_delta').reset_index()    
        within_minute_matched = within_minute.loc[0]
        matching_events.append([within_minute_matched['time_delta'], within_minute_matched['frame'],
                                row.event_id, within_minute_matched['event_time']])
    del within_minute

  first_matching = pd.DataFrame(matching_events,columns=['time_delta','frame_model','event_id','event_time_model'])

  min_deltas = first_matching.groupby(['frame_model'])['time_delta'].transform(min)
  first_matching = first_matching.loc[first_matching['time_delta'] == min_deltas].reset_index(drop=True)
  first_matching['wave'] = 'first'


  second_matching = []

  tags_second = ['block','duel','tackle','aerial duel','take on against','physical duel']                
  candidate = all_ev.loc[all_ev['event_name'].isin(tags_second)]

  leftovers_model = passes_model_open.loc[~passes_model_open['frame'].isin(first_matching['frame_model'].tolist()),
                                     ['event_time','event_name','jersey_team','frame','player_id','time_ms']]         

  thres2 = thres/2                                                     
  for row in leftovers_model.itertuples():

    t0 = row.time_ms - thres2*1000
    t1 = row.time_ms + thres2*1000

    within_minute = candidate.loc[(candidate['time_ms'] >= t0) &
                                  (candidate['time_ms'] <= t1)].copy()
    if within_minute.empty:
      continue
    else:
      within_minute['same_player'] = within_minute['player_1'].apply(match_player_interval,args=(row.player_id,))
      within_minute['time_delta'] = within_minute['time_ms'].apply(lambda x: abs(x - row.time_ms)/factor)
      within_minute = within_minute.loc[(within_minute['same_player'] == True) & (within_minute['time_delta'] < thres2)]

      if within_minute.empty:
        continue
      else:
        within_minute = within_minute.sort_values(by='time_delta').reset_index()    
        within_minute_matched = within_minute.loc[0]
        second_matching.append([within_minute_matched['time_delta'], row.frame,
                                within_minute_matched['event_id'], row.event_time,'second'])


  second_matching = pd.DataFrame(second_matching,columns=['time_delta','frame_model','event_id','event_time_model','wave'])

  min_deltas = second_matching.groupby(['event_id'])['time_delta'].transform(min)
  second_matching = second_matching.loc[second_matching['time_delta'] == min_deltas].reset_index(drop=True)


  ## Join passes and set pieces. Set piece passes are also counted in passes due to the way they are tagged
  all_ev = official_ev['all']
  events_match = pd.concat([first_matching, second_matching, match_dbe], ignore_index=True).sort_values(by='frame_model')
  events_match = events_match.merge(all_ev[['event_time','event_name','from_set_piece','jersey_team_1',
                                            'event_id','event_outcome']], how='left', on='event_id')
  events_match = events_match.merge(passes_model[['frame','event_name','from_set_piece','dead_ball_event','step_change']].rename(columns={'frame':'frame_model'}), how='left', on='frame_model',suffixes=('','_model'))
  
  finally_unmatched_model = passes_model.loc[~passes_model['frame'].isin(events_match['frame_model'].unique()),'frame']
  finally_unmatched_model = passes_model[['event_time','event_name','from_set_piece',
                                        'jersey_team','player_id','frame',
                                        'dead_ball_event','step_change']].merge(finally_unmatched_model, how='right', on='frame')
  finally_unmatched_model['source'] = 'model'
  finally_unmatched_model.rename(columns={'frame':'event_id'}, inplace=True)

  passes_dbe = pd.concat([passes_ev, dbe_ev], ignore_index=True)
  finally_unmatched_off = passes_dbe.loc[~passes_dbe['event_id'].isin(events_match['event_id'].unique()),'event_id']
  finally_unmatched_off = passes_dbe[['event_time','event_name','from_set_piece','jersey_team_1',
                                     'player_1','event_id','event_outcome']].merge(finally_unmatched_off, how='right', on='event_id')

  finally_unmatched_off['source'] = 'official'
  finally_unmatched_off = finally_unmatched_off.rename(columns={'jersey_team_1':'jersey_team',
                                                                'player_1':'player_id'})


  unmatched_events = pd.concat([finally_unmatched_model,finally_unmatched_off],ignore_index=True)
  unmatched_events = unmatched_events.sort_values(by='event_time',ascending=True).reset_index(drop=True)


  ## Matching duels
  duels_ev = official_ev['duels']
  duels_model = model.query("contest.str.contains('duel')", engine='python').copy().reset_index().rename(columns={'index':'duel_index'})
  contest_model = model.query("contest == 'contest'", engine='python').copy().reset_index().rename(columns={'index':'duel_index'})
  contest_duels_model = model.query("contest != ''", engine='python').copy().reset_index().rename(columns={'index':'duel_index'})
  duel_filter = 5
  matching_duels = []
  for row in duels_ev.itertuples():
    check_contest = False
    t0 = row.time_ms - duel_filter*1000
    t1 = row.time_ms + duel_filter*1000

    within_minute = duels_model.loc[(duels_model['time_ms'] >= t0) &
                                        (duels_model['time_ms'] <= t1)].copy()
    
    if within_minute.empty:
      check_contest = True
    else:
      within_minute['same_player_1'] = within_minute['contest_player_id'].apply(match_player_interval,args=(row.player_1,))
      within_minute['time_delta'] = abs(within_minute['time_ms'] - row.time_ms)/1000
      within_minute = within_minute.loc[(within_minute['same_player_1'] == True) & \
                                        (within_minute['time_delta'] < duel_filter)]

      if within_minute.empty:
        check_contest = True
      else:
        within_minute = within_minute.sort_values(by='time_delta').reset_index()    
        within_minute_matched = within_minute.loc[0]
        matching_duels.append([within_minute_matched['time_delta'], within_minute_matched['frame'],
                                row.event_id, 
                                within_minute_matched['event_time'], 'duel',
                                1 if 'E' in within_minute_matched['step_change'] else 0])
    del within_minute
    if check_contest:
      contest_check = contest_model.loc[(contest_model['time_ms'] >= t0) &
                                        (contest_model['time_ms'] <= t1)].copy()
    
      if contest_check.empty:
        continue
      else:
        contest_check['same_player_1'] = contest_check['contest_player_id'].apply(match_player_interval,args=(row.player_1,))
        contest_check['time_delta'] = abs(contest_check['time_ms'] - row.time_ms)/1000
        contest_check = contest_check.loc[(contest_check['same_player_1'] == True) & \
                                          (contest_check['time_delta'] < duel_filter)]

        if contest_check.empty:
          continue
        else:
          contest_check = contest_check.sort_values(by='time_delta').reset_index()    
          contest_check_matched = contest_check.loc[0]
          matching_duels.append([contest_check_matched['time_delta'], contest_check_matched['frame'],
                                  row.event_id, contest_check_matched['event_time'], 'contest',
                                  1 if 'E' in contest_check_matched['step_change'] else 0])
    

  duels_match = pd.DataFrame(matching_duels,columns=['time_delta','frame','event_id','event_time_model','wave','step'])
  duels_match = duels_match.merge(duels_ev[['event_time','event_name','jersey_team_1','event_id']], how='left', on='event_id')
  duels_match = duels_match.merge(contest_duels_model[['frame','contest','contest_jersey_team']].rename(columns={'contest':'event_name'}), how='left', on='frame',suffixes=('','_model'))
  
  unmatched_duels_model = duels_model.loc[~duels_model['frame'].isin(duels_match['frame'].unique()),'frame']
  unmatched_duels_model = duels_model[['event_time','contest',
                                       'contest_player_id',
                                       'frame']].merge(unmatched_duels_model, how='right', on='frame').rename(columns={'contest':'event_name'})
  unmatched_duels_model['jersey_team'] = unmatched_duels_model['contest_player_id'].apply(jersey_team, args=(params,))
  unmatched_duels_model['source'] = 'model'
  unmatched_duels_model.drop(columns=['contest_player_id'], inplace=True)
  unmatched_duels_model.rename(columns={'frame':'event_id'}, inplace=True)

  unmatched_duels_off = duels_ev.loc[~duels_ev['event_id'].isin(duels_match['event_id'].unique()),'event_id']
  unmatched_duels_off = duels_ev[['event_time','jersey_team_1',
                                   'event_id','event_name']].merge(unmatched_duels_off, how='right', on='event_id').rename(columns={'jersey_team_1':'jersey_team'})

  unmatched_duels_off['source'] = 'official'
  
  unmatched_duels = pd.concat([unmatched_duels_model,unmatched_duels_off],ignore_index=True)
  unmatched_duels = unmatched_duels.sort_values(by='event_time',ascending=True).reset_index(drop=True)
  duels_match.drop(columns=['time_delta','event_id'],inplace=True)


  ## Matching goals
  goals_ev = all_ev.query("event_name == 'goal' or event_name == 'own goal'").reset_index(drop=True)
  goals_model_raw = model.query("dead_ball_event == 'goal'").copy().reset_index().rename(columns={'index':'goal_index'})
  if not goals_model_raw.empty:
    ii = []
    ev = []
    for row in goals_model_raw.itertuples():

      j = model.loc[:row.goal_index,'event_name'].str.contains("|".join(['shot','own','cross']))[::-1].idxmax()
      ii.append(j)
      ev.append('own goal' if model.at[j,'event_name'] == 'own goal' else 'goal')
    goals_model = model.loc[ii].copy().reset_index(drop=True)
    goals_model['dbe'] = ev
    
  else:
    goals_model = goals_model_raw.copy()
    goals_model['dbe'] = goals_model['dead_ball_event'] 

  goal_filter = 15
  matching_goals = []
  for row in goals_ev.itertuples():

    t0 = row.time_ms - goal_filter*1000
    t1 = row.time_ms + goal_filter*1000

    within_minute = goals_model.loc[(goals_model['time_ms'] >= t0) &
                                        (goals_model['time_ms'] <= t1)].copy()
    
    if within_minute.empty:
      continue
    else:
      if len(set(within_minute.iloc[0]['team_id'].split(';'))) == 2:
        within_minute['same_team'] = True
      else:
        if (row.event_name == 'goal' and within_minute.iloc[0]['dbe'] == 'goal') or (row.event_name == 'own goal' and within_minute.iloc[0]['dbe'] == 'own goal'):        
          within_minute['same_team'] = within_minute['team_id'].apply(match_player_interval,args=(row.team_1,))
        else:
          within_minute['same_team'] = ~(within_minute['team_id'].apply(match_player_interval,args=(row.team_1,)))
      # within_minute['same_player'] = within_minute['player_id'].apply(match_player_interval,args=(row.player_1,))
      within_minute['time_delta'] = within_minute['event_time'].apply(lambda x: abs(x.to_event_time()[1] - row.time_ms)/1000)
      within_minute = within_minute.loc[(within_minute['same_team'] == True) & (within_minute['time_delta'] < goal_filter)]

      if within_minute.empty:
        continue
      else:
        within_minute = within_minute.sort_values(by='time_delta').reset_index()    
        within_minute_matched = within_minute.loc[0]
        matching_goals.append([within_minute_matched['time_delta'], within_minute_matched['frame'],
                                row.event_id, within_minute_matched['event_time']])
    del within_minute

  goals_match = pd.DataFrame(matching_goals,columns=['time_delta','frame','event_id','event_time_model'])
  goals_match = goals_match.merge(all_ev[['event_time','event_name',
                                          'jersey_team_1','event_id']].rename(columns={'jersey_team_1':'jersey_team'}), how='left', on='event_id')
  goals_match = goals_match.merge(goals_model[['frame','event_name','dbe','jersey_team']], how='left', on='frame',suffixes=('','_model'))
  
  unmatched_goals_model = goals_model.loc[~goals_model['frame'].isin(goals_match['frame'].unique()),'frame']
  unmatched_goals_model = goals_model[['event_time','event_name','jersey_team',
                                       'player_id','frame','dbe']].merge(unmatched_goals_model, how='right', on='frame')
  unmatched_goals_model['source'] = 'model'
  unmatched_goals_model.rename(columns={'frame':'event_id'}, inplace=True)

  unmatched_goals_off = goals_ev.loc[~goals_ev['event_id'].isin(goals_match['event_id'].unique()),'event_id']
  unmatched_goals_off = goals_ev[['event_time','event_name','jersey_team_1',
                                     'player_1','event_id']].merge(unmatched_goals_off, how='right', on='event_id')

  unmatched_goals_off['source'] = 'official'
  unmatched_goals_off = unmatched_goals_off.rename(columns={'jersey_team_1':'jersey_team',
                                                                'player_1':'player_id'})
  unmatched_goals = pd.concat([unmatched_goals_model,unmatched_goals_off],ignore_index=True)
  unmatched_goals = unmatched_goals.sort_values(by='event_time',ascending=True).reset_index(drop=True)
  goals_match.drop(columns=['event_id'],inplace=True)


  ## Matching saves
  saves_ev = official_ev['saves']

  saves_model = final_model.query("is_goalkeeper == True and event_name.str.contains('|'.join(['save','claim','reception']))", engine='python').reset_index(drop=True)
  saves_only = final_model.query("is_goalkeeper == True and event_name.str.contains('|'.join(['save','claim']))", engine='python').reset_index(drop=True)
  saves_first = []
  save_filter = 10
  for row in saves_ev.itertuples():

    t0 = row.time_ms - save_filter*1000
    t1 = row.time_ms + save_filter*1000

    within_minute = saves_model.loc[(saves_model['time_ms'] >= t0) &
                                      (saves_model['time_ms'] <= t1)].copy()
    if within_minute.empty:
      continue
    else:
      within_minute['same_player'] = within_minute['player_id'].apply(match_player_interval,args=(row.player_1,))  
      within_minute['time_delta'] = within_minute['event_time'].apply(lambda x: abs(x.to_event_time()[1] - row.time_ms)/1000)
      within_minute = within_minute.loc[(within_minute['same_player'] == True) & (within_minute['time_delta'] < save_filter)]

      if within_minute.empty:
        continue
      else:
        within_minute = within_minute.sort_values(by='time_delta').reset_index()    
        within_minute_matched = within_minute.loc[0]
        saves_first.append([within_minute_matched['time_delta'], within_minute_matched['frame'],
                                row.event_id, within_minute_matched['event_time']])
    del within_minute

  saves_first = pd.DataFrame(saves_first,columns=['time_delta','frame','event_id','event_time_model'])
  min_deltas = saves_first.groupby(['frame'])['time_delta'].transform(min)
  saves_match = saves_first.loc[saves_first['time_delta'] == min_deltas].reset_index(drop=True)
  saves_match['wave'] = 'first'

  
  saves_match = saves_match.merge(pd.concat([saves_ev, all_ev], ignore_index=True)[['event_time','event_name','jersey_team_1','event_id','event_outcome']], how='left', on='event_id')
  saves_match = saves_match.merge(saves_model[['frame','event_name']], how='left', on='frame',suffixes=('','_model'))

  unmatched_saves_model = saves_only.loc[(~saves_only['frame'].isin(saves_match['frame'].unique())),'frame']
  unmatched_saves_model = saves_only[['event_time','event_name','from_set_piece','jersey_team','player_id','frame']].merge(unmatched_saves_model, how='right', on='frame')
  unmatched_saves_model['source'] = 'model'
  unmatched_saves_model.rename(columns={'frame':'event_id'}, inplace=True)

  unmatched_saves_off = saves_ev.loc[~saves_ev['event_id'].isin(saves_match['event_id'].unique()),'event_id']
  unmatched_saves_off = saves_ev[['event_time','event_name','from_set_piece','jersey_team_1',
                                     'player_1','event_id','action_type']].merge(unmatched_saves_off, how='right', on='event_id')

  unmatched_saves_off['source'] = 'official'
  unmatched_saves_off = unmatched_saves_off.rename(columns={'jersey_team_1':'jersey_team',
                                                                            'player_1':'player_id'})

  unmatched_saves = pd.concat([unmatched_saves_model,unmatched_saves_off],ignore_index=True)
  unmatched_saves = unmatched_saves.sort_values(by='event_time',ascending=True).reset_index(drop=True)

  saves_match.drop(columns=['time_delta','event_id','frame'], inplace=True)


  ## Confusion matrices
  results = dict()
  results['open_play'] = dict()  
  results['pass_shot'] = dict()  
  results['pass_shot_cross'] = dict()    
  results['duel'] = dict()
  results['save'] = dict()
  results['goal'] = dict()
  results['set_piece'] = dict()

  matching = dict()
  matching['passes_match'] = events_match
  matching['passes_unmatch'] = unmatched_events
  matching['saves_match'] = saves_match
  matching['saves_unmatch'] = unmatched_saves
  matching['goals_match'] = goals_match
  matching['goals_unmatch'] = unmatched_goals
  matching['duels_match'] = duels_match
  matching['duels_unmatch'] = unmatched_duels


  ## Passes/shots/crosses including those from set pieces, although not the set piece events per se
  pass_names = ['pass/shot/cross','none']
  events_match_binary = events_match.query("event_name.str.contains('|'.join(@pass_tags))", engine='python').reset_index(drop=True)
  pm_binary = np.zeros((len(pass_names),len(pass_names)))
  pm_binary[0,0] = events_match_binary.shape[0]
  pm_binary[0,1] = finally_unmatched_off.shape[0]
  pm_binary[1,0] = finally_unmatched_model.shape[0]
  results['open_play']['names'] = pass_names
  results['open_play']['performance'] = pm_binary     

  shot_tags_matching = ['shot']
  pass_tags_matching = [tag for tag in pass_tags if tag not in ['cross']+shot_tags_matching]

  pm_open = np.zeros((4,4))
  pm_open[0,0] = events_match_binary.query("event_name.isin(@pass_tags_matching) and event_name_model == 'pass'").shape[0]
  pm_open[1,1] = events_match_binary.query("event_name.isin(@shot_tags_matching) and event_name_model.str.contains('shot')").shape[0]  
  pm_open[2,2] = events_match_binary.query("event_name == 'cross' and event_name_model == 'cross'").shape[0] 
  pm_open[0,1] = events_match_binary.query("event_name.isin(@pass_tags_matching) and event_name_model.str.contains('shot')").shape[0]
  pm_open[0,2] = events_match_binary.query("event_name.isin(@pass_tags_matching) and event_name_model == 'cross'").shape[0]
  pm_open[1,0] = events_match_binary.query("event_name.isin(@shot_tags_matching) and event_name_model == 'pass'").shape[0]
  pm_open[1,2] = events_match_binary.query("event_name.isin(@shot_tags_matching) and event_name_model == 'cross'").shape[0]
  pm_open[2,0] = events_match_binary.query("event_name == 'cross' and event_name_model == 'pass'").shape[0]
  pm_open[2,1] = events_match_binary.query("event_name == 'cross' and event_name_model.str.contains('shot')").shape[0]
  pm_open[-1,0] = finally_unmatched_model.query("event_name == 'pass'").shape[0]
  pm_open[-1,1] = finally_unmatched_model.query("event_name.str.contains('shot')").shape[0]
  pm_open[-1,2] = finally_unmatched_model.query("event_name == 'cross'").shape[0]
  pm_open[0,-1] = finally_unmatched_off.query("event_name.isin(@pass_tags_matching)").shape[0]
  pm_open[1,-1] = finally_unmatched_off.query("event_name.isin(@shot_tags_matching)").shape[0]
  pm_open[2,-1] = finally_unmatched_off.query("event_name == 'cross'").shape[0]
  results['pass_shot_cross']['names'] = ['pass','shot','cross','none']
  results['pass_shot_cross']['performance'] = pm_open

  pass_tags_matching = [tag for tag in pass_tags if tag not in shot_tags_matching]
  pm_open = np.zeros((3,3))
  pm_open[0,0] = events_match_binary.query("event_name.isin(@pass_tags_matching) and event_name_model.isin(['pass','cross'])").shape[0]
  pm_open[1,1] = events_match_binary.query("event_name.isin(@shot_tags_matching) and event_name_model.str.contains('shot')").shape[0]  
  pm_open[0,1] = events_match_binary.query("event_name.isin(@pass_tags_matching) and event_name_model.str.contains('shot')").shape[0]
  pm_open[1,0] = events_match_binary.query("event_name.isin(@shot_tags_matching) and event_name_model.isin(['pass','cross'])").shape[0]
  pm_open[-1,0] = finally_unmatched_model.query("event_name.isin(['pass','cross'])").shape[0]
  pm_open[-1,1] = finally_unmatched_model.query("event_name.str.contains('shot')").shape[0]
  pm_open[0,-1] = finally_unmatched_off.query("event_name.isin(@pass_tags_matching)").shape[0]
  pm_open[1,-1] = finally_unmatched_off.query("event_name.isin(@shot_tags_matching)").shape[0]
  results['pass_shot']['names'] = ['pass/cross','shot','none']
  results['pass_shot']['performance'] = pm_open
    
  ## Saves
  save_names = ['save','claim','loose?','none']
  pm_save = np.zeros((len(save_names), len(save_names)))
  if not saves_ev.empty:      
      pm_save[0,0] = saves_match.query("event_name == 'goal prevention' and event_name_model == 'save'", engine='python').shape[0]
      pm_save[0,1] = saves_match.query("event_name == 'goal prevention' and event_name_model == 'claim'", engine='python').shape[0]
      pm_save[0,2] = saves_match.query("event_name == 'goal prevention' and event_name_model.str.contains('reception')", engine='python').shape[0]
      pm_save[0,-1] = unmatched_saves_off.query("event_name == 'goal prevention'").shape[0]

      pm_save[1,0] = saves_match.query("event_name == 'aerial control' and event_name_model == 'save'", engine='python').shape[0]
      pm_save[1,1] = saves_match.query("event_name == 'aerial control' and event_name_model == 'claim'", engine='python').shape[0]
      pm_save[1,2] = saves_match.query("event_name == 'aerial control' and event_name_model.str.contains('reception')", engine='python').shape[0]
      pm_save[1,-1] = unmatched_saves_off.query("event_name == 'aerial control'", engine='python').shape[0]

      pm_save[-1,0] = unmatched_saves_model.query("event_name == 'save'", engine='python').shape[0]
      pm_save[-1,1] = unmatched_saves_model.query("event_name == 'claim'", engine='python').shape[0]

  results['save']['names'] = save_names
  results['save']['performance'] = pm_save
  
  ## Goals
  goal_names = ['goal','own goal','none']

  pm_goal = np.zeros((len(goal_names),len(goal_names)))
  pm_goal[0,0] = goals_match.query("event_name == 'goal' and dbe == 'goal'", engine='python').shape[0]
  pm_goal[0,1] = goals_match.query("event_name == 'goal' and dbe == 'own goal'", engine='python').shape[0]
  pm_goal[1,0] = goals_match.query("event_name == 'own goal' and dbe == 'goal'", engine='python').shape[0]
  pm_goal[1,1] = goals_match.query("event_name == 'own goal' and dbe == 'own goal'", engine='python').shape[0]
  pm_goal[0,-1] = unmatched_goals_off.query("event_name != 'own goal'").shape[0]
  pm_goal[1,-1] = unmatched_goals_off.query("event_name == 'own goal'").shape[0]
  pm_goal[-1,0] = unmatched_goals_model.query("dbe == 'goal'").shape[0]
  pm_goal[-1,1] = unmatched_goals_model.query("dbe == 'own goal'").shape[0]

  results['goal']['names'] = goal_names
  results['goal']['performance'] = pm_goal

  ## Duels
  duel_names = ['air duel','ground duel','contest','none']
  duel_dcu_names = ['aerial duel','physical duel','duel','tackle','take on against','none']

  pm_duel = np.zeros((len(duel_dcu_names),len(duel_names)))
  pm_duel[0,0] = duels_match.query("event_name == 'aerial duel' and event_name_model == 'air duel'", engine='python').shape[0]
  pm_duel[0,1] = duels_match.query("event_name == 'aerial duel' and event_name_model == 'ground duel'", engine='python').shape[0]
  pm_duel[0,2] = duels_match.query("event_name == 'aerial duel' and event_name_model == 'contest'", engine='python').shape[0]
  pm_duel[1,0] = duels_match.query("event_name == 'physical duel' and event_name_model == 'air duel'", engine='python').shape[0]  
  pm_duel[1,1] = duels_match.query("event_name == 'physical duel' and event_name_model == 'ground duel'", engine='python').shape[0]
  pm_duel[1,2] = duels_match.query("event_name == 'physical duel' and event_name_model == 'contest'", engine='python').shape[0]
  pm_duel[2,0] = duels_match.query("event_name == 'duel' and event_name_model == 'air duel'", engine='python').shape[0]  
  pm_duel[2,1] = duels_match.query("event_name == 'duel' and event_name_model == 'ground duel'", engine='python').shape[0]
  pm_duel[2,2] = duels_match.query("event_name == 'duel' and event_name_model == 'contest'", engine='python').shape[0]
  pm_duel[3,0] = duels_match.query("event_name == 'tackle' and event_name_model == 'air duel'", engine='python').shape[0]
  pm_duel[3,1] = duels_match.query("event_name == 'tackle' and event_name_model == 'ground duel'", engine='python').shape[0]
  pm_duel[3,2] = duels_match.query("event_name == 'tackle' and event_name_model == 'contest'", engine='python').shape[0]
  pm_duel[4,0] = duels_match.query("event_name == 'take on against' and event_name_model == 'air duel'", engine='python').shape[0]
  pm_duel[4,1] = duels_match.query("event_name == 'take on against' and event_name_model == 'ground duel'", engine='python').shape[0]
  pm_duel[4,2] = duels_match.query("event_name == 'take on against' and event_name_model == 'contest'", engine='python').shape[0]
  pm_duel[0,-1] = unmatched_duels_off.query("event_name == 'aerial duel'").shape[0]
  pm_duel[1,-1] = unmatched_duels_off.query("event_name == 'physical duel'").shape[0]
  pm_duel[2,-1] = unmatched_duels_off.query("event_name == 'duel'").shape[0]
  pm_duel[3,-1] = unmatched_duels_off.query("event_name == 'tackle'").shape[0]
  pm_duel[4,-1] = unmatched_duels_off.query("event_name == 'take on against'").shape[0]
  pm_duel[-1,0] = unmatched_duels_model.query("event_name == 'air duel'").shape[0]
  pm_duel[-1,1] = unmatched_duels_model.query("event_name == 'ground duel'").shape[0]

  results['duel']['names'] = (duel_dcu_names, duel_names)
  results['duel']['performance'] = pm_duel


  ## Duels again, but now omitting the secondary duels in the auto-events data, that is those where there was a possession loss upon running
  ## possession_model but later got erased due to player continuation

  duels_ev = official_ev['duels']
  duels_model = model.query("contest.str.contains('duel') and ~step_change.str.contains('17')", engine='python').copy().reset_index().rename(columns={'index':'duel_index'})
  contest_model = model.query("contest == 'contest'", engine='python').copy().reset_index().rename(columns={'index':'duel_index'})
  contest_duels_model = model.query("contest != ''", engine='python').copy().reset_index().rename(columns={'index':'duel_index'})
  duel_filter = 5
  matching_duels = []
  for row in duels_ev.itertuples():
    check_contest = False
    t0 = row.time_ms - duel_filter*1000
    t1 = row.time_ms + duel_filter*1000

    within_minute = duels_model.loc[(duels_model['time_ms'] >= t0) &
                                        (duels_model['time_ms'] <= t1)].copy()
    
    if within_minute.empty:
      check_contest = True
    else:
      within_minute['same_player_1'] = within_minute['contest_player_id'].apply(match_player_interval,args=(row.player_1,))
      within_minute['time_delta'] = abs(within_minute['time_ms'] - row.time_ms)/1000
      within_minute = within_minute.loc[(within_minute['same_player_1'] == True) & \
                                        (within_minute['time_delta'] < duel_filter)]

      if within_minute.empty:
        check_contest = True
      else:
        within_minute = within_minute.sort_values(by='time_delta').reset_index()    
        within_minute_matched = within_minute.loc[0]
        matching_duels.append([within_minute_matched['time_delta'], within_minute_matched['frame'],
                                row.event_id, 
                                within_minute_matched['event_time'], 'duel',
                                1 if 'E' in within_minute_matched['step_change'] else 0])
    del within_minute
    if check_contest:
      contest_check = contest_model.loc[(contest_model['time_ms'] >= t0) &
                                        (contest_model['time_ms'] <= t1)].copy()
    
      if contest_check.empty:
        continue
      else:
        contest_check['same_player_1'] = contest_check['contest_player_id'].apply(match_player_interval,args=(row.player_1,))
        contest_check['time_delta'] = abs(contest_check['time_ms'] - row.time_ms)/1000
        contest_check = contest_check.loc[(contest_check['same_player_1'] == True) & \
                                          (contest_check['time_delta'] < duel_filter)]

        if contest_check.empty:
          continue
        else:
          contest_check = contest_check.sort_values(by='time_delta').reset_index()    
          contest_check_matched = contest_check.loc[0]
          matching_duels.append([contest_check_matched['time_delta'], contest_check_matched['frame'],
                                  row.event_id, contest_check_matched['event_time'], 'contest',
                                  1 if 'E' in contest_check_matched['step_change'] else 0])
    

  duels_match = pd.DataFrame(matching_duels,columns=['time_delta','frame','event_id','event_time_model','wave','step'])
  duels_match = duels_match.merge(duels_ev[['event_time','event_name','jersey_team_1','event_id']], how='left', on='event_id')
  duels_match = duels_match.merge(contest_duels_model[['frame','contest','contest_jersey_team']].rename(columns={'contest':'event_name'}), how='left', on='frame',suffixes=('','_model'))
  
  unmatched_duels_model = duels_model.loc[~duels_model['frame'].isin(duels_match['frame'].unique()),'frame']
  unmatched_duels_model = duels_model[['event_time','contest',
                                       'contest_player_id',
                                       'frame']].merge(unmatched_duels_model, how='right', on='frame').rename(columns={'contest':'event_name'})
  unmatched_duels_model['jersey_team'] = unmatched_duels_model['contest_player_id'].apply(jersey_team, args=(params,))
  unmatched_duels_model['source'] = 'model'
  unmatched_duels_model.drop(columns=['contest_player_id'], inplace=True)
  unmatched_duels_model.rename(columns={'frame':'event_id'}, inplace=True)

  unmatched_duels_off = duels_ev.loc[~duels_ev['event_id'].isin(duels_match['event_id'].unique()),'event_id']
  unmatched_duels_off = duels_ev[['event_time','jersey_team_1',
                                   'event_id','event_name']].merge(unmatched_duels_off, how='right', on='event_id').rename(columns={'jersey_team_1':'jersey_team'})

  unmatched_duels_off['source'] = 'official'
  
  unmatched_duels = pd.concat([unmatched_duels_model,unmatched_duels_off],ignore_index=True)
  unmatched_duels = unmatched_duels.sort_values(by='event_time',ascending=True).reset_index(drop=True)
  duels_match.drop(columns=['time_delta','event_id'],inplace=True)

  pm_duel = np.zeros((len(duel_dcu_names),len(duel_names)))
  pm_duel[0,0] = duels_match.query("event_name == 'aerial duel' and event_name_model == 'air duel'", engine='python').shape[0]
  pm_duel[0,1] = duels_match.query("event_name == 'aerial duel' and event_name_model == 'ground duel'", engine='python').shape[0]
  pm_duel[0,2] = duels_match.query("event_name == 'aerial duel' and event_name_model == 'contest'", engine='python').shape[0]
  pm_duel[1,0] = duels_match.query("event_name == 'physical duel' and event_name_model == 'air duel'", engine='python').shape[0]  
  pm_duel[1,1] = duels_match.query("event_name == 'physical duel' and event_name_model == 'ground duel'", engine='python').shape[0]
  pm_duel[1,2] = duels_match.query("event_name == 'physical duel' and event_name_model == 'contest'", engine='python').shape[0]
  pm_duel[2,0] = duels_match.query("event_name == 'duel' and event_name_model == 'air duel'", engine='python').shape[0]  
  pm_duel[2,1] = duels_match.query("event_name == 'duel' and event_name_model == 'ground duel'", engine='python').shape[0]
  pm_duel[2,2] = duels_match.query("event_name == 'duel' and event_name_model == 'contest'", engine='python').shape[0]
  pm_duel[3,0] = duels_match.query("event_name == 'tackle' and event_name_model == 'air duel'", engine='python').shape[0]
  pm_duel[3,1] = duels_match.query("event_name == 'tackle' and event_name_model == 'ground duel'", engine='python').shape[0]
  pm_duel[3,2] = duels_match.query("event_name == 'tackle' and event_name_model == 'contest'", engine='python').shape[0]
  pm_duel[4,0] = duels_match.query("event_name == 'take on against' and event_name_model == 'air duel'", engine='python').shape[0]
  pm_duel[4,1] = duels_match.query("event_name == 'take on against' and event_name_model == 'ground duel'", engine='python').shape[0]
  pm_duel[4,2] = duels_match.query("event_name == 'take on against' and event_name_model == 'contest'", engine='python').shape[0]
  pm_duel[0,-1] = unmatched_duels_off.query("event_name == 'aerial duel'").shape[0]
  pm_duel[1,-1] = unmatched_duels_off.query("event_name == 'physical duel'").shape[0]
  pm_duel[2,-1] = unmatched_duels_off.query("event_name == 'duel'").shape[0]
  pm_duel[3,-1] = unmatched_duels_off.query("event_name == 'tackle'").shape[0]
  pm_duel[4,-1] = unmatched_duels_off.query("event_name == 'take on against'").shape[0]
  pm_duel[-1,0] = unmatched_duels_model.query("event_name == 'air duel'").shape[0]
  pm_duel[-1,1] = unmatched_duels_model.query("event_name == 'ground duel'").shape[0]

  results['duel_first'] = dict()
  results['duel_first']['names'] = (duel_dcu_names, duel_names)
  results['duel_first']['performance'] = pm_duel



  ## Set pieces
  events_match_setpiece = events_match.query("wave == 'dbe'", engine='python').reset_index(drop=True)
  results_origin = dict()
  set_names = ['kickoff','penalty kick','corner kick','throw-in','goal kick','free kick','free kick?','none']
  pm_set = np.zeros((len(set_names),len(set_names)))
  for i, name_i in enumerate(set_names[:-1]):
    for j, name_j in enumerate(set_names[:-1]):
        pm_set[i,j] = events_match_setpiece.loc[(events_match_setpiece['from_set_piece'] == name_i) & (events_match_setpiece['from_set_piece_model'] == name_j)].shape[0]
  
  for j, name_j in enumerate(set_names[:-1]):
    pm_set[-1,j] = finally_unmatched_model.query("from_set_piece == @name_j", engine='python').shape[0]
  
  for i, name_i in enumerate(set_names[:-1]):
    pm_set[i,-1] = finally_unmatched_off.query("from_set_piece == @name_i", engine='python').shape[0]

  results['set_piece']['names'] = set_names
  results['set_piece']['performance'] = pm_set
  results['pass_tags'] = pass_tags


  return results, matching



def matching_kognia_against_dcu(kog, dcu, thres, half):
  """
    This routine compares the Kognia events with the DCU events

    :param kog: Dataframe of Kognia events
    :param dcu: Dataframe of DCU events
    :param thres: Threshold for matching passes
    :param half: Half of game
    :return: Dictionary with the performance metrics and with actual matched events for matching passes, set pieces, saves, duels
  """ 

  factor = 1000

  dcu_pass_tags = ['pass','shot','interception','cross','clearance','block']
  dcu_duel_tags = ['tackle','aerial duel','duel','physical duel','take on against']
  dcu_action_tags = ['miscontrol','unintentional_touch']
  kog_pass_tags = ['pass','shot','interception','uncontrolled']
  kog_duel_tags = ['tackle']
    
  ## Set pieces
  kog_dbe = kog.query("from_set_piece.notnull() and from_set_piece != 'kickoff start'").reset_index(drop=True)
  dcu_dbe = dcu.query("from_set_piece.notnull() and from_set_piece != 'kickoff start'").reset_index(drop=True)
  kog_dbe_purged = kog_dbe.copy()

  first_filter = 15
  matching_dbe = []
  for row in dcu_dbe.itertuples():

    t0 = row.time - first_filter*1000
    t1 = row.time + first_filter*1000

    within_minute = kog_dbe_purged.loc[(kog_dbe_purged['time'] >= t0) &
                                        (kog_dbe_purged['time'] <= t1)].copy()
    
    if within_minute.empty:
      continue
    else:
      within_minute['same_team'] = within_minute['team_id'] == row.team_id
      within_minute['time_delta'] = within_minute['time'].apply(lambda x: abs(x - row.time)/1000)
      within_minute = within_minute.loc[within_minute['same_team'] == True]

      if within_minute.empty:
        continue
      else:
        within_minute = within_minute.sort_values(by='time_delta').reset_index()   
        within_minute_matched_dbe = within_minute.loc[0]
        matching_dbe.append([within_minute_matched_dbe['time_delta'], 
                             within_minute_matched_dbe['event_id'],
                            within_minute_matched_dbe['time'],
                             row.event_id, row.time])
        kog_dbe_purged.drop(index=within_minute.at[0,'index'], inplace=True)

    del within_minute

  match_dbe = pd.DataFrame(matching_dbe,columns=['time_delta','id_kog','time_kog','id_dcu','time_dcu'])

  min_deltas = match_dbe.groupby(['time_kog'])['time_delta'].transform(min)
  match_dbe = match_dbe.loc[match_dbe['time_delta'] == min_deltas].reset_index(drop=True)
  match_dbe['wave'] = 'dbe'

  ## Passes
  kog_passes = kog.query("event_name.isin(@kog_pass_tags) or event_name.isin(@kog_duel_tags)", engine='python').reset_index(drop=True)
  dcu_passes = dcu.query("event.isin(@dcu_pass_tags) or event.isin(@dcu_duel_tags) or action_type.isin(@dcu_action_tags)", engine='python').reset_index(drop=True)
  kog_passes_purged = kog_passes.copy()

  matching_events = []
  for row in dcu_passes.itertuples():

    t0 = row.time + thres[0]*1000
    t1 = row.time + thres[1]*1000
        
    within_minute = kog_passes_purged.loc[(kog_passes_purged['time'] >= t0) &
                                          (kog_passes_purged['time'] <= t1)].copy()
    if within_minute.empty:
      continue
    else:
      within_minute['same_player'] = within_minute['player_id'] == row.from_player_id
      within_minute['time_delta'] = within_minute['time'].apply(lambda x: abs(x - row.time)/1000)
      within_minute = within_minute.loc[(within_minute['same_player'] == True)]# & (within_minute['time_delta'] < thres)]

      if within_minute.empty:
        continue
      else:
        if row.event in dcu_pass_tags:
            within_minute_pass = within_minute.query("event_name.isin(@kog_pass_tags)")
            if within_minute_pass.empty:
                matching_events, index_drop = add_match(within_minute, matching_events, row)
            else:
                matching_events, index_drop = add_match(within_minute_pass, matching_events, row)
        elif row.event in dcu_duel_tags:
            within_minute_duel = within_minute.query("event_name.isin(@kog_duel_tags)")
            if within_minute_duel.empty:
                matching_events, index_drop = add_match(within_minute, matching_events, row)
            else:
                matching_events, index_drop = add_match(within_minute_duel, matching_events, row)
        else:
            matching_events, index_drop = add_match(within_minute, matching_events, row)

        kog_passes_purged.drop(index=index_drop, inplace=True)

    del within_minute

  first_matching = pd.DataFrame(matching_events,columns=['time_delta','id_kog','time_kog','id_dcu','time_dcu'])

  min_deltas = first_matching.groupby(['time_kog'])['time_delta'].transform(min)
  first_matching = first_matching.loc[first_matching['time_delta'] == min_deltas].reset_index(drop=True)
  first_matching['wave'] = 'first'
 

  dcu_passes_dbe = pd.concat([dcu_passes, dcu_dbe], ignore_index=True).drop_duplicates(subset='event_id')
  kog_passes_dbe = pd.concat([kog_passes, kog_dbe], ignore_index=True).drop_duplicates(subset='event_id')

  events_match = pd.concat([first_matching, match_dbe], ignore_index=True).sort_values(by='time_dcu')
  events_match = events_match.merge(dcu_passes_dbe[['event','from_player_shirt_number','team_name',
                                                    'from_set_piece','action_type',
                                          'event_id']].rename(columns={'event_id':'id_dcu',
                                                                       'from_player_shirt_number':'jersey',
                                                                      'team_name':'team',
                                                                      'action_type':'action_dcu'}), how='left', on='id_dcu')
  events_match = events_match.merge(kog_passes_dbe[['event_id','event_name','from_set_piece']].rename(columns={'event_id':'id_kog','event_name':'event'}), how='left', 
                                    on='id_kog',suffixes=('_dcu','_kog'))
  
  pass_unmatched_kog = kog_passes_dbe.loc[~kog_passes_dbe['event_id'].isin(events_match['id_kog'].unique()),'event_id']
  pass_unmatched_kog = kog_passes_dbe[['time','event_name','from_set_piece',
                                        'jersey','player_id','event_id','team']].merge(pass_unmatched_kog, how='right', on='event_id')
  pass_unmatched_kog['source'] = 'kognia'
  pass_unmatched_kog.rename(columns={'event_name':'event'}, inplace=True)

  pass_unmatched_dcu = dcu_passes_dbe.loc[~dcu_passes_dbe['event_id'].isin(events_match['id_dcu'].unique()),'event_id']
  pass_unmatched_dcu = dcu_passes_dbe[['time','event','from_set_piece','from_player_shirt_number',
                                     'from_player_id','event_id','team_name','action_type']].merge(pass_unmatched_dcu, how='right', on='event_id')

  pass_unmatched_dcu['source'] = 'dcu'
  pass_unmatched_dcu = pass_unmatched_dcu.rename(columns={'from_player_shirt_number':'jersey',
                                                                'from_player_id':'player_id',
                                                         'team_name':'team','action_type':'action_dcu'})

 
  unmatched_events = pd.concat([pass_unmatched_kog,
                                pass_unmatched_dcu],ignore_index=True)
  unmatched_events = unmatched_events.sort_values(by='time',ascending=True).reset_index(drop=True)
  unmatched_events['game_time'] = unmatched_events['time'].apply(lambda x: GameTime.from_event_time(half, x))

  ## Goals
  dcu_goals = dcu.query("event == 'goal' or event == 'own goal'").reset_index(drop=True)
  kog_goals = kog.query("event_name == 'goal' or event_name == 'own goal'").reset_index(drop=True)
    
  goal_filter = 30
  matching_goals = []
  for row in dcu_goals.itertuples():

    t0 = row.time - goal_filter*1000
    t1 = row.time + goal_filter*1000

    within_minute = kog_goals.loc[(kog_goals['time'] >= t0) &
                                        (kog_goals['time'] <= t1)].copy()
    
    if within_minute.empty:
      continue
    else:
      within_minute['same_player'] = within_minute['player_id'] == row.from_player_id
      within_minute['time_delta'] = within_minute['time'].apply(lambda x: abs(x - row.time)/1000)
      within_minute = within_minute.loc[(within_minute['same_player'] == True)]# & (within_minute['time_delta'] < goal_filter)]

      if within_minute.empty:
        continue
      else:
        within_minute = within_minute.sort_values(by='time_delta').reset_index()    
        within_minute_matched = within_minute.loc[0]
        matching_goals.append([within_minute_matched['time_delta'], 
                              within_minute_matched['event_id'],
                              within_minute_matched['time'],
                                row.event_id, within_minute_matched['time']])
    del within_minute

  goals_match = pd.DataFrame(matching_goals,columns=['time_delta','id_kog','time_kog','id_dcu','time_dcu'])
  goals_match = goals_match.merge(dcu[['event','from_player_shirt_number','team_name',
                                                    'from_set_piece',
                                          'event_id']].rename(columns={'event_id':'id_dcu',
                                                                       'from_player_shirt_number':'jersey',
                                                                      'team_name':'team'}), how='left', on='id_dcu')
  goals_match = goals_match.merge(kog[['event_id','event_name']].rename(columns={'event_id':'id_kog','event_name':'event'}), 
                                  how='left', on='id_kog',suffixes=('_dcu','_kog'))
  
  goals_unmatched_kog = kog_goals.loc[~kog_goals['event_id'].isin(goals_match['id_kog'].unique()),'event_id']
  goals_unmatched_kog = kog_goals[['time','event_name','from_set_piece',
                                        'jersey','player_id','event_id','team']].merge(goals_unmatched_kog, how='right', on='event_id')
  goals_unmatched_kog['source'] = 'kog'
  goals_unmatched_kog.rename(columns={'event_name':'event'}, inplace=True)

  goals_unmatched_dcu = dcu_goals.loc[~dcu_goals['event_id'].isin(goals_match['id_dcu'].unique()),'event_id']
  goals_unmatched_dcu = dcu_goals[['time','event','from_set_piece','from_player_shirt_number',
                                     'from_player_id','event_id','team_name']].merge(goals_unmatched_dcu, how='right', on='event_id')

  goals_unmatched_dcu['source'] = 'dcu'
  goals_unmatched_dcu = goals_unmatched_dcu.rename(columns={'from_player_shirt_number':'jersey',
                                                                'from_player_id':'player_id',
                                                           'team_name':'team'})

  unmatched_goals = pd.concat([goals_unmatched_kog,goals_unmatched_dcu],ignore_index=True)
  unmatched_goals = unmatched_goals.sort_values(by='time',ascending=True).reset_index(drop=True)
  unmatched_goals['game_time'] = unmatched_goals['time'].apply(lambda x: GameTime.from_event_time(half, x))


  ## Fouls
  dcu_fouls = dcu.query("event == 'foul against' or event == 'offside'").reset_index(drop=True)
  kog_fouls = kog.query("event_name == 'foul' or event_name == 'offside'").reset_index(drop=True)
    
  foul_filter = 10
  matching_fouls = []
  for row in dcu_fouls.itertuples():

    t0 = row.time - foul_filter*1000
    t1 = row.time + foul_filter*1000

    within_minute = kog_fouls.loc[(kog_fouls['time'] >= t0) &
                                  (kog_fouls['time'] <= t1)].copy()
    
    if within_minute.empty:
      continue
    else:
      within_minute['same_team'] = within_minute['team_id'] == row.team_id
      within_minute['time_delta'] = within_minute['time'].apply(lambda x: abs(x - row.time)/1000)
      within_minute = within_minute.loc[(within_minute['same_team'] == True)]

      if within_minute.empty:
        continue
      else:
        within_minute = within_minute.sort_values(by='time_delta').reset_index()    
        within_minute_matched = within_minute.loc[0]
        matching_fouls.append([within_minute_matched['time_delta'], 
                              within_minute_matched['event_id'],
                              within_minute_matched['time'],
                                row.event_id, within_minute_matched['time']])
    del within_minute

  fouls_match = pd.DataFrame(matching_fouls,columns=['time_delta','id_kog','time_kog','id_dcu','time_dcu'])
  fouls_match = fouls_match.merge(dcu[['event','team_name',
                                          'event_id']].rename(columns={'event_id':'id_dcu',
                                                                      'team_name':'team'}), how='left', on='id_dcu')
  fouls_match = fouls_match.merge(kog[['event_id','event_name']].rename(columns={'event_id':'id_kog','event_name':'event'}), 
                                  how='left', on='id_kog',suffixes=('_dcu','_kog'))
  
  fouls_unmatched_kog = kog_fouls.loc[~kog_fouls['event_id'].isin(fouls_match['id_kog'].unique()),'event_id']
  fouls_unmatched_kog = kog_fouls[['time','event_name','event_id','team']].merge(fouls_unmatched_kog, how='right', on='event_id')
  fouls_unmatched_kog['source'] = 'kog'
  fouls_unmatched_kog.rename(columns={'event_name':'event'}, inplace=True)

  fouls_unmatched_dcu = dcu_fouls.loc[~dcu_fouls['event_id'].isin(fouls_match['id_dcu'].unique()),'event_id']
  fouls_unmatched_dcu = dcu_fouls[['time','event','event_id','team_name']].merge(fouls_unmatched_dcu, how='right', on='event_id')

  fouls_unmatched_dcu['source'] = 'dcu'
  fouls_unmatched_dcu = fouls_unmatched_dcu.rename(columns={'team_name':'team'})

  unmatched_fouls = pd.concat([fouls_unmatched_kog,fouls_unmatched_dcu],ignore_index=True)
  unmatched_fouls = unmatched_fouls.sort_values(by='time',ascending=True).reset_index(drop=True)
  unmatched_fouls['game_time'] = unmatched_fouls['time'].apply(lambda x: GameTime.from_event_time(half, x))


  ## Confusion matrices
  results = dict()
  results['open_play'] = dict()  
  results['pass_shot'] = dict()  
  results['pass_shot_cross'] = dict()    
  results['goal'] = dict()
  results['set_piece'] = dict()
  results['foul'] = dict()

  matching = dict()
  matching['passes_match'] = events_match
  matching['passes_unmatch'] = unmatched_events
  matching['goals_match'] = goals_match
  matching['goals_unmatch'] = unmatched_goals
  matching['fouls_match'] = fouls_match
  matching['fouls_unmatch'] = unmatched_fouls

  ## Passes/shots/crosses including those from set pieces, although not the set piece events per se
  pass_names = ['pass/shot/cross','none']
  events_match_binary = events_match
  pm_binary = np.zeros((len(pass_names),len(pass_names)))
  pm_binary[0,0] = events_match_binary.shape[0]
  pm_binary[0,1] = pass_unmatched_dcu.shape[0]
  pm_binary[1,0] = pass_unmatched_kog.shape[0]
  results['open_play']['names'] = pass_names
  results['open_play']['performance'] = pm_binary     


  kog_open_names = kog_pass_tags + kog_duel_tags + ['none']
  dcu_open_names = dcu_pass_tags + dcu_action_tags + dcu_duel_tags + ['none']
  pm_open = np.zeros((len(dcu_open_names),len(kog_open_names)))
  for i,dcu_name in enumerate(dcu_open_names[:-1]):
    if dcu_name not in dcu_action_tags:
        pm_open[i,-1] = pass_unmatched_dcu.query("event == @dcu_name").shape[0]
    else:
        pm_open[i,-1] = pass_unmatched_dcu.query("action_dcu == @dcu_name").shape[0]

    for j, kog_name in enumerate(kog_open_names[:-1]):
        if i == 0:
            pm_open[-1,j] = pass_unmatched_kog.query("event == @kog_name").shape[0]

        if dcu_name not in dcu_action_tags:
            pm_open[i,j] = events_match.query("event_dcu == @dcu_name and event_kog == @kog_name").shape[0]
        else:
            pm_open[i,j] = events_match.query("action_dcu == @dcu_name and event_kog == @kog_name").shape[0]


  results['pass_shot_cross']['names'] = (dcu_open_names, kog_open_names)
  results['pass_shot_cross']['performance'] = pm_open

  
  ## Goals
  goal_names = ['goal','own goal','none']

  pm_goal = np.zeros((len(goal_names),len(goal_names)))
  pm_goal[0,0] = goals_match.query("event_dcu == 'goal' and event_kog == 'goal'", engine='python').shape[0]
  pm_goal[0,1] = goals_match.query("event_dcu == 'goal' and event_kog == 'own goal'", engine='python').shape[0]
  pm_goal[1,0] = goals_match.query("event_dcu == 'own goal' and event_kog == 'goal'", engine='python').shape[0]
  pm_goal[1,1] = goals_match.query("event_dcu == 'own goal' and event_kog == 'own goal'", engine='python').shape[0]
  pm_goal[0,-1] = goals_unmatched_dcu.query("event == 'goal'").shape[0]
  pm_goal[1,-1] = goals_unmatched_dcu.query("event == 'own goal'").shape[0]
  pm_goal[-1,0] = goals_unmatched_kog.query("event == 'goal'").shape[0]
  pm_goal[-1,1] = goals_unmatched_kog.query("event == 'own goal'").shape[0]

  results['goal']['names'] = goal_names
  results['goal']['performance'] = pm_goal

  ## Fouls
  foul_names = ['foul','offside','none']

  pm_foul = np.zeros((len(foul_names),len(foul_names)))
  pm_foul[0,0] = fouls_match.query("event_dcu == 'foul against' and event_kog == 'foul'", engine='python').shape[0]
  pm_foul[0,1] = fouls_match.query("event_dcu == 'foul against' and event_kog == 'offside'", engine='python').shape[0]
  pm_foul[1,0] = fouls_match.query("event_dcu == 'offside' and event_kog == 'foul'", engine='python').shape[0]
  pm_foul[1,1] = fouls_match.query("event_dcu == 'offside' and event_kog == 'offside'", engine='python').shape[0]
  pm_foul[0,-1] = fouls_unmatched_dcu.query("event == 'foul against'").shape[0]
  pm_foul[1,-1] = fouls_unmatched_dcu.query("event == 'offside'").shape[0]
  pm_foul[-1,0] = fouls_unmatched_kog.query("event == 'foul'").shape[0]
  pm_foul[-1,1] = fouls_unmatched_kog.query("event == 'offside'").shape[0]

  results['foul']['names'] = foul_names
  results['foul']['performance'] = pm_foul

  ## Set pieces
  events_match_setpiece = events_match.query("wave == 'dbe'", engine='python').reset_index(drop=True)
  results_origin = dict()
  set_names = ['kickoff','penalty kick','corner kick','throw-in','goal kick','free kick','none']
  pm_set = np.zeros((len(set_names),len(set_names)))
  for i, name_i in enumerate(set_names[:-1]):
    for j, name_j in enumerate(set_names[:-1]):
        pm_set[i,j] = events_match_setpiece.loc[(events_match_setpiece['from_set_piece_dcu'] == name_i) & (events_match_setpiece['from_set_piece_kog'] == name_j)].shape[0]
  
  for j, name_j in enumerate(set_names[:-1]):
    pm_set[-1,j] = pass_unmatched_kog.query("from_set_piece == @name_j", engine='python').shape[0]
  
  for i, name_i in enumerate(set_names[:-1]):
    pm_set[i,-1] = pass_unmatched_dcu.query("from_set_piece == @name_i", engine='python').shape[0]

  results['set_piece']['names'] = set_names
  results['set_piece']['performance'] = pm_set
  results['dcu_pass_tags'] = dcu_pass_tags
  results['kog_pass_tags'] = kog_pass_tags


  return results, matching


def add_match(within_minute, matching_events, row):

    within_minute = within_minute.sort_values(by='time_delta').reset_index()    
    within_minute_matched = within_minute.loc[0]
    matching_events.append([within_minute_matched['time_delta'], 
                          within_minute_matched['event_id'],
                          within_minute_matched['time'],
                          row.event_id, row.time])

    return matching_events, within_minute.at[0,'index']

def matching_kognia_against_sts(kog, sts, thres, half):
  """
    This routine compares the Kognia events with the STS events

    :param kog: Dataframe of Kognia events
    :param sts: Dataframe of STS events
    :param thres: Threshold for matching passes
    :param half: Half of game
    :return: Dictionary with the performance metrics and with actual matched events for matching passes, set pieces, saves, duels
  """ 

  factor = 1000

  sts_pass_tags = ['pass','shot','cross','clearance','other ball action']
  sts_duel_tags = ['duel']
  kog_pass_tags = ['pass','shot','interception','uncontrolled']
  kog_duel_tags = ['tackle']

  ## Set pieces
  kog_dbe = kog.query("from_set_piece.notnull() and from_set_piece != 'kickoff start'").reset_index(drop=True)
  sts_dbe = sts.query("from_set_piece.notnull() and from_set_piece != 'kickoff start'").reset_index(drop=True)
  kog_dbe_purged = kog_dbe.copy()

  first_filter = 15
  matching_dbe = []
  for row in sts_dbe.itertuples():

    t0 = row.time - first_filter*1000
    t1 = row.time + first_filter*1000

    within_minute = kog_dbe_purged.loc[(kog_dbe_purged['time'] >= t0) &
                                        (kog_dbe_purged['time'] <= t1)].copy()
    
    if within_minute.empty:
      # unmatched.append([row['event_id'],'official','no data'])
      continue
    else:
      within_minute['same_team'] = within_minute['team_id'] == row.team_1
      within_minute['time_delta'] = within_minute['time'].apply(lambda x: abs(x - row.time)/1000)
      within_minute = within_minute.loc[within_minute['same_team'] == True]

      if within_minute.empty:
        continue
      else:              
        within_minute = within_minute.sort_values(by='time_delta').reset_index()    
        within_minute_matched_dbe = within_minute.loc[0]
        matching_dbe.append([within_minute_matched_dbe['time_delta'], 
                             within_minute_matched_dbe['event_id'],
                            within_minute_matched_dbe['time'],
                             row.event_id, row.time])
        kog_dbe_purged.drop(index=within_minute.at[0,'index'], inplace=True)

    del within_minute

  match_dbe = pd.DataFrame(matching_dbe,columns=['time_delta','id_kog','time_kog','id_sts','time_sts'])

  min_deltas = match_dbe.groupby(['time_kog'])['time_delta'].transform(min)
  match_dbe = match_dbe.loc[match_dbe['time_delta'] == min_deltas].reset_index(drop=True)
  match_dbe['wave'] = 'dbe'

  ## Passes
  kog_passes = kog.query("event_name.isin(@kog_pass_tags) or event_name.isin(@kog_duel_tags)", engine='python').reset_index(drop=True)
  sts_passes = sts.query("event_name.isin(@sts_pass_tags) or event_name.isin(@sts_duel_tags)", engine='python').reset_index(drop=True)
  kog_passes_purged = kog_passes.copy()

  matching_events = []
  for row in sts_passes.itertuples():

    t0 = row.time + thres[0]*1000
    t1 = row.time + thres[1]*1000
        
    within_minute = kog_passes_purged.loc[(kog_passes_purged['time'] >= t0) &
                                          (kog_passes_purged['time'] <= t1)].copy()
    if within_minute.empty:
      continue
    else:
      within_minute['same_player'] = within_minute['player_id'] == row.player_1
      within_minute['time_delta'] = within_minute['time'].apply(lambda x: abs(x - row.time)/1000)
      within_minute = within_minute.loc[(within_minute['same_player'] == True)]# & (within_minute['time_delta'] < thres)]

      if within_minute.empty:
        continue
      else:
        if row.event_name in sts_pass_tags:
            within_minute_pass = within_minute.query("event_name.isin(@kog_pass_tags)")
            if within_minute_pass.empty:
                matching_events, index_drop = add_match(within_minute, matching_events, row)
            else:
                matching_events, index_drop = add_match(within_minute_pass, matching_events, row)
        elif row.event_name in sts_duel_tags:
            within_minute_duel = within_minute.query("event_name.isin(@kog_duel_tags)")
            if within_minute_duel.empty:
                matching_events, index_drop = add_match(within_minute, matching_events, row)
            else:
                matching_events, index_drop = add_match(within_minute_duel, matching_events, row)
        
        # kog_passes_purged.drop(index=index_drop, inplace=True)

    del within_minute

  first_matching = pd.DataFrame(matching_events,columns=['time_delta','id_kog','time_kog','id_sts','time_sts'])

  min_deltas = first_matching.groupby(['time_kog'])['time_delta'].transform(min)
  first_matching = first_matching.loc[first_matching['time_delta'] == min_deltas].reset_index(drop=True)
  first_matching['wave'] = 'first'
 

  sts_passes_dbe = pd.concat([sts_passes, sts_dbe], ignore_index=True).drop_duplicates(subset='event_id')
  kog_passes_dbe = pd.concat([kog_passes, kog_dbe], ignore_index=True).drop_duplicates(subset='event_id')

  events_match = pd.concat([first_matching, match_dbe], ignore_index=True).sort_values(by='time_sts')
  events_match = events_match.merge(sts_passes_dbe[['event_name','jersey_1','team_name_1',
                                                    'from_set_piece',
                                          'event_id']].rename(columns={'event_id':'id_sts',
                                                                       'jersey_1':'jersey',
                                                                      'team_name_1':'team','event_name':'event'}), how='left', on='id_sts')
  events_match = events_match.merge(kog_passes_dbe[['event_id','event_name','from_set_piece']].rename(columns={'event_id':'id_kog','event_name':'event'}), how='left', 
                                    on='id_kog',suffixes=('_sts','_kog'))
  
  pass_unmatched_kog = kog_passes_dbe.loc[~kog_passes_dbe['event_id'].isin(events_match['id_kog'].unique()),'event_id']
  pass_unmatched_kog = kog_passes_dbe[['time','event_name','from_set_piece',
                                        'jersey','player_id','event_id','team']].merge(pass_unmatched_kog, how='right', on='event_id')
  pass_unmatched_kog['source'] = 'kognia'
  pass_unmatched_kog.rename(columns={'event_name':'event'}, inplace=True)

  pass_unmatched_sts = sts_passes_dbe.loc[~sts_passes_dbe['event_id'].isin(events_match['id_sts'].unique()),'event_id']
  pass_unmatched_sts = sts_passes_dbe[['time','event_name','from_set_piece','jersey_1',
                                     'player_1','event_id','team_name_1']].merge(pass_unmatched_sts, how='right', on='event_id')

  pass_unmatched_sts['source'] = 'sts'
  pass_unmatched_sts = pass_unmatched_sts.rename(columns={'jersey_1':'jersey',
                                                                'player_1':'player_id',
                                                         'team_name_1':'team','event_name':'event'})

 
  unmatched_events = pd.concat([pass_unmatched_kog,
                                pass_unmatched_sts],ignore_index=True)
  unmatched_events = unmatched_events.sort_values(by='time',ascending=True).reset_index(drop=True)
  unmatched_events['game_time'] = unmatched_events['time'].apply(lambda x: GameTime.from_event_time(half, x))

  ## Goals
  sts_goals = sts.query("event_name == 'goal' or event_name == 'own goal'").reset_index(drop=True)
  kog_goals = kog.query("event_name == 'goal' or event_name == 'own goal'").reset_index(drop=True)
    
  goal_filter = 30
  matching_goals = []
  for row in sts_goals.itertuples():

    t0 = row.time - goal_filter*1000
    t1 = row.time + goal_filter*1000

    within_minute = kog_goals.loc[(kog_goals['time'] >= t0) &
                                        (kog_goals['time'] <= t1)].copy()
    
    if within_minute.empty:
      continue
    else:
      within_minute['same_player'] = within_minute['player_id'] == row.player_1
      within_minute['time_delta'] = within_minute['time'].apply(lambda x: abs(x - row.time)/1000)
      within_minute = within_minute.loc[(within_minute['same_player'] == True)]# & (within_minute['time_delta'] < goal_filter)]

      if within_minute.empty:
        continue
      else:
        within_minute = within_minute.sort_values(by='time_delta').reset_index()    
        within_minute_matched = within_minute.loc[0]
        matching_goals.append([within_minute_matched['time_delta'], 
                              within_minute_matched['event_id'],
                              within_minute_matched['time'],
                                row.event_id, row.time])
    del within_minute

  goals_match = pd.DataFrame(matching_goals,columns=['time_delta','id_kog','time_kog','id_sts','time_sts'])
  goals_match = goals_match.merge(sts[['event_name','jersey_1','team_name_1',
                                                    'from_set_piece',
                                          'event_id']].rename(columns={'event_id':'id_sts',
                                                                       'jersey_1':'jersey',
                                                                      'team_name_1':'team',
                                                                      'event_name':'event'}), how='left', on='id_sts')
  goals_match = goals_match.merge(kog[['event_id','event_name']].rename(columns={'event_id':'id_kog','event_name':'event'}), 
                                  how='left', on='id_kog',suffixes=('_sts','_kog'))
  
  goals_unmatched_kog = kog_goals.loc[~kog_goals['event_id'].isin(goals_match['id_kog'].unique()),'event_id']
  goals_unmatched_kog = kog_goals[['time','event_name','from_set_piece',
                                        'jersey','player_id','event_id','team']].merge(goals_unmatched_kog, how='right', on='event_id')
  goals_unmatched_kog['source'] = 'kog'
  goals_unmatched_kog.rename(columns={'event_name':'event'}, inplace=True)

  goals_unmatched_sts = sts_goals.loc[~sts_goals['event_id'].isin(goals_match['id_sts'].unique()),'event_id']
  goals_unmatched_sts = sts_goals[['time','event_name','from_set_piece','jersey_1',
                                     'player_1','event_id','team_name_1']].merge(goals_unmatched_sts, how='right', on='event_id')

  goals_unmatched_sts['source'] = 'sts'
  goals_unmatched_sts = goals_unmatched_sts.rename(columns={'jersey_1':'jersey',
                                                                'player_1':'player_id',
                                                           'team_name_1':'team',
                                                           'event_name':'event'})

  unmatched_goals = pd.concat([goals_unmatched_kog,goals_unmatched_sts],ignore_index=True)
  unmatched_goals = unmatched_goals.sort_values(by='time',ascending=True).reset_index(drop=True)
  unmatched_goals['game_time'] = unmatched_goals['time'].apply(lambda x: GameTime.from_event_time(half, x))


  ## Confusion matrices
  results = dict()
  results['open_play'] = dict()  
  results['pass_shot'] = dict()  
  results['pass_shot_cross'] = dict()    
  results['goal'] = dict()
  results['set_piece'] = dict()

  matching = dict()
  matching['passes_match'] = events_match
  matching['passes_unmatch'] = unmatched_events
  matching['goals_match'] = goals_match
  matching['goals_unmatch'] = unmatched_goals

  ## Passes/shots/crosses including those from set pieces, although not the set piece events per se
  pass_names = ['pass/shot/cross','none']
  events_match_binary = events_match
  pm_binary = np.zeros((len(pass_names),len(pass_names)))
  pm_binary[0,0] = events_match_binary.shape[0]
  pm_binary[0,1] = pass_unmatched_sts.shape[0]
  pm_binary[1,0] = pass_unmatched_kog.shape[0]
  results['open_play']['names'] = pass_names
  results['open_play']['performance'] = pm_binary     


  kog_open_names = kog_pass_tags + kog_duel_tags + ['none']
  sts_open_names = sts_pass_tags + sts_duel_tags + ['none']
  pm_open = np.zeros((len(sts_open_names),len(kog_open_names)))
  for i,sts_name in enumerate(sts_open_names[:-1]):
    pm_open[i,-1] = pass_unmatched_sts.query("event == @sts_name").shape[0]
    
    for j, kog_name in enumerate(kog_open_names[:-1]):
        if i == 0:
            pm_open[-1,j] = pass_unmatched_kog.query("event == @kog_name").shape[0]

        pm_open[i,j] = events_match.query("event_sts == @sts_name and event_kog == @kog_name").shape[0]
        
  results['pass_shot_cross']['names'] = (sts_open_names, kog_open_names)
  results['pass_shot_cross']['performance'] = pm_open

    
  
  ## Goals
  goal_names = ['goal','own goal','none']

  pm_goal = np.zeros((len(goal_names),len(goal_names)))
  pm_goal[0,0] = goals_match.query("event_sts == 'goal' and event_kog == 'goal'", engine='python').shape[0]
  pm_goal[0,1] = goals_match.query("event_sts == 'goal' and event_kog == 'own goal'", engine='python').shape[0]
  pm_goal[1,0] = goals_match.query("event_sts == 'own goal' and event_kog == 'goal'", engine='python').shape[0]
  pm_goal[1,1] = goals_match.query("event_sts == 'own goal' and event_kog == 'own goal'", engine='python').shape[0]
  pm_goal[0,-1] = goals_unmatched_sts.query("event == 'goal'").shape[0]
  pm_goal[1,-1] = goals_unmatched_sts.query("event == 'own goal'").shape[0]
  pm_goal[-1,0] = goals_unmatched_kog.query("event == 'goal'").shape[0]
  pm_goal[-1,1] = goals_unmatched_kog.query("event == 'own goal'").shape[0]

  results['goal']['names'] = goal_names
  results['goal']['performance'] = pm_goal


  ## Set pieces
  events_match_setpiece = events_match.query("wave == 'dbe'", engine='python').reset_index(drop=True)
  results_origin = dict()
  set_names = ['kickoff','penalty kick','corner kick','throw-in','goal kick','free kick','none']
  pm_set = np.zeros((len(set_names),len(set_names)))
  for i, name_i in enumerate(set_names[:-1]):
    for j, name_j in enumerate(set_names[:-1]):
        pm_set[i,j] = events_match_setpiece.loc[(events_match_setpiece['from_set_piece_sts'] == name_i) & (events_match_setpiece['from_set_piece_kog'] == name_j)].shape[0]
  
  for j, name_j in enumerate(set_names[:-1]):
    pm_set[-1,j] = pass_unmatched_kog.query("from_set_piece == @name_j", engine='python').shape[0]
  
  for i, name_i in enumerate(set_names[:-1]):
    pm_set[i,-1] = pass_unmatched_sts.query("from_set_piece == @name_i", engine='python').shape[0]

  results['set_piece']['names'] = set_names
  results['set_piece']['performance'] = pm_set
  results['sts_pass_tags'] = sts_pass_tags
  results['kog_pass_tags'] = kog_pass_tags


  return results, matching

def matching_sts_against_dcu(sts, dcu, thres, half):
  """
    This routine compares the STS events with the DCU events

    :param sts: Dataframe of STS events
    :param dcu: Dataframe of DCU events
    :param thres: Threshold for matching passes
    :param half: Half of game
    :return: Dictionary with the performance metrics and with actual matched events for matching passes, set pieces, saves, duels
  """ 

  factor = 1000

  dcu_pass_tags = ['shot','pass','cross','clearance','own goal']
  sts_pass_tags = ['shot','pass','cross','clearance','own goal','fair play','goal']
    
  ## Set pieces
  sts_dbe = sts.query("from_set_piece.notnull() and from_set_piece != 'kickoff start'").reset_index(drop=True)
  dcu_dbe = dcu.query("from_set_piece.notnull() and from_set_piece != 'kickoff start'").reset_index(drop=True)

  first_filter = 15
  matching_dbe = []
  for row in dcu_dbe.itertuples():

    t0 = row.time - first_filter*1000
    t1 = row.time + first_filter*1000

    within_minute = sts_dbe.loc[(sts_dbe['time'] >= t0) &
                                (sts_dbe['time'] <= t1)].copy()
    
    if within_minute.empty:
      # unmatched.append([row['event_id'],'official','no data'])
      continue
    else:
      within_minute['same_team'] = within_minute['team_1'] == row.team_id
      within_minute['time_delta'] = within_minute['time'].apply(lambda x: abs(x - row.time)/1000)
      within_minute = within_minute.loc[within_minute['same_team'] == True]

      if within_minute.empty:
        continue
      else:
        within_minute = within_minute.sort_values(by='time_delta').reset_index()    
        within_minute_matched_dbe = within_minute.loc[0]
        matching_dbe.append([within_minute_matched_dbe['time_delta'], 
                             within_minute_matched_dbe['event_id'],
                            within_minute_matched_dbe['time'],
                             row.event_id, row.time])
    del within_minute

  match_dbe = pd.DataFrame(matching_dbe,columns=['time_delta','id_sts','time_sts','id_dcu','time_dcu'])

  min_deltas = match_dbe.groupby(['time_sts'])['time_delta'].transform(min)
  match_dbe = match_dbe.loc[match_dbe['time_delta'] == min_deltas].reset_index(drop=True)
  match_dbe['wave'] = 'dbe'

  ## Pases
  sts_passes = sts.query("event_name.isin(@sts_pass_tags)", engine='python').reset_index(drop=True)
  dcu_passes = dcu.query("event.isin(@dcu_pass_tags)", engine='python').reset_index(drop=True)

  matching_events = []
  for row in dcu_passes.itertuples():

    t0 = row.time + thres[0]*1000
    t1 = row.time + thres[1]*1000
        
    within_minute = sts_passes.loc[(sts_passes['time'] >= t0) &
                                          (sts_passes['time'] <= t1)].copy()
    if within_minute.empty:
      continue
    else:
      within_minute['same_player'] = within_minute['player_1'] == row.from_player_id
      within_minute['time_delta'] = within_minute['time'].apply(lambda x: abs(x - row.time)/1000)
      within_minute = within_minute.loc[(within_minute['same_player'] == True)]# & (within_minute['time_delta'] < thres)]

      if within_minute.empty:
        continue
      else:
        within_minute = within_minute.sort_values(by='time_delta').reset_index()    
        within_minute_matched = within_minute.loc[0]
        matching_events.append([within_minute_matched['time_delta'], 
                              within_minute_matched['event_id'],
                              within_minute_matched['time'],
                                row.event_id, within_minute_matched['time']])
    del within_minute

  first_matching = pd.DataFrame(matching_events,columns=['time_delta','id_sts','time_sts','id_dcu','time_dcu'])

  min_deltas = first_matching.groupby(['time_sts'])['time_delta'].transform(min)
  first_matching = first_matching.loc[first_matching['time_delta'] == min_deltas].reset_index(drop=True)
  first_matching['wave'] = 'first'
 

  dcu_passes_dbe = pd.concat([dcu_passes, dcu_dbe], ignore_index=True).drop_duplicates(subset='event_id')
  sts_passes_dbe = pd.concat([sts_passes, sts_dbe], ignore_index=True).drop_duplicates(subset='event_id')

  events_match = pd.concat([first_matching, match_dbe], ignore_index=True).sort_values(by='time_dcu')
  events_match = events_match.merge(dcu_passes_dbe[['event','from_player_shirt_number','team_name',
                                                    'from_set_piece',
                                          'event_id']].rename(columns={'event_id':'id_dcu',
                                                                       'from_player_shirt_number':'jersey',
                                                                      'team_name':'team'}), how='left', on='id_dcu')
  events_match = events_match.merge(sts_passes_dbe[['event_id','event_name','from_set_piece']].rename(columns={'event_id':'id_sts','event_name':'event'}), how='left', 
                                    on='id_sts',suffixes=('_dcu','_sts'))
  
  pass_unmatched_sts = sts_passes_dbe.loc[~sts_passes_dbe['event_id'].isin(events_match['id_sts'].unique()),'event_id']
  pass_unmatched_sts = sts_passes_dbe[['time','event_name','from_set_piece',
                                        'jersey_1','player_1','event_id','team_name_1']].merge(pass_unmatched_sts, how='right', on='event_id')
  pass_unmatched_sts['source'] = 'sts'
  pass_unmatched_sts.rename(columns={'event_name':'event','jersey_1':'jersey',
                                        'player_1':'player_id','team_name_1':'team'}, inplace=True)

  pass_unmatched_dcu = dcu_passes_dbe.loc[~dcu_passes_dbe['event_id'].isin(events_match['id_dcu'].unique()),'event_id']
  pass_unmatched_dcu = dcu_passes_dbe[['time','event','from_set_piece','from_player_shirt_number',
                                     'from_player_id','event_id','team_name']].merge(pass_unmatched_dcu, how='right', on='event_id')

  pass_unmatched_dcu['source'] = 'dcu'
  pass_unmatched_dcu = pass_unmatched_dcu.rename(columns={'from_player_shirt_number':'jersey',
                                                                'from_player_id':'player_id',
                                                         'team_name':'team'})

 
  unmatched_events = pd.concat([pass_unmatched_sts,
                                pass_unmatched_dcu],ignore_index=True)
  unmatched_events = unmatched_events.sort_values(by='time',ascending=True).reset_index(drop=True)
  unmatched_events['game_time'] = unmatched_events['time'].apply(lambda x: GameTime.from_event_time(half, x))

  ## Goals
  dcu_goals = dcu.query("event == 'goal' or event == 'own goal'").reset_index(drop=True)
  sts_goals = sts.query("event_name == 'goal' or event_name == 'own goal'").reset_index(drop=True)
    
  goal_filter = 30
  matching_goals = []
  for row in dcu_goals.itertuples():

    t0 = row.time - goal_filter*1000
    t1 = row.time + goal_filter*1000

    within_minute = sts_goals.loc[(sts_goals['time'] >= t0) &
                                        (sts_goals['time'] <= t1)].copy()
    
    if within_minute.empty:
      continue
    else:
      within_minute['same_player'] = within_minute['player_1'] == row.from_player_id
      within_minute['time_delta'] = within_minute['time'].apply(lambda x: abs(x - row.time)/1000)
      within_minute = within_minute.loc[(within_minute['same_player'] == True)]# & (within_minute['time_delta'] < goal_filter)]

      if within_minute.empty:
        continue
      else:
        within_minute = within_minute.sort_values(by='time_delta').reset_index()    
        within_minute_matched = within_minute.loc[0]
        matching_goals.append([within_minute_matched['time_delta'], 
                              within_minute_matched['event_id'],
                              within_minute_matched['time'],
                                row.event_id, within_minute_matched['time']])
    del within_minute

  goals_match = pd.DataFrame(matching_goals,columns=['time_delta','id_sts','time_sts','id_dcu','time_dcu'])
  goals_match = goals_match.merge(dcu[['event','from_player_shirt_number','team_name',
                                                    'from_set_piece',
                                          'event_id']].rename(columns={'event_id':'id_dcu',
                                                                       'from_player_shirt_number':'jersey',
                                                                      'team_name':'team'}), how='left', on='id_dcu')
  goals_match = goals_match.merge(sts[['event_id','event_name']].rename(columns={'event_id':'id_sts','event_name':'event'}), 
                                  how='left', on='id_sts',suffixes=('_dcu','_sts'))
  
  goals_unmatched_sts = sts_goals.loc[~sts_goals['event_id'].isin(goals_match['id_sts'].unique()),'event_id']
  goals_unmatched_sts = sts_goals[['time','event_name','from_set_piece',
                                        'jersey_1','player_1','event_id','team_name_1']].merge(goals_unmatched_sts, how='right', on='event_id')
  goals_unmatched_sts['source'] = 'sts'
  goals_unmatched_sts.rename(columns={'event_name':'event','jersey_1':'jersey',
                                        'player_1':'player_id','team_name_1':'team'}, inplace=True)

  goals_unmatched_dcu = dcu_goals.loc[~dcu_goals['event_id'].isin(goals_match['id_dcu'].unique()),'event_id']
  goals_unmatched_dcu = dcu_goals[['time','event','from_set_piece','from_player_shirt_number',
                                     'from_player_id','event_id','team_name']].merge(goals_unmatched_dcu, how='right', on='event_id')

  goals_unmatched_dcu['source'] = 'dcu'
  goals_unmatched_dcu = goals_unmatched_dcu.rename(columns={'from_player_shirt_number':'jersey',
                                                                'from_player_id':'player_id',
                                                           'team_name':'team'})

  unmatched_goals = pd.concat([goals_unmatched_sts,goals_unmatched_dcu],ignore_index=True)
  unmatched_goals = unmatched_goals.sort_values(by='time',ascending=True).reset_index(drop=True)
  unmatched_goals['game_time'] = unmatched_goals['time'].apply(lambda x: GameTime.from_event_time(half, x))


  ## Confusion matrices
  results = dict()
  results['open_play'] = dict()  
  results['pass_shot'] = dict()  
  results['pass_shot_cross'] = dict()    
  results['goal'] = dict()
  results['set_piece'] = dict()

  matching = dict()
  matching['passes_match'] = events_match
  matching['passes_unmatch'] = unmatched_events
  matching['goals_match'] = goals_match
  matching['goals_unmatch'] = unmatched_goals

  ## Passes/shots/crosses including those from set pieces, although not the set piece events per se
  pass_names = ['pass/attempt/cross','none']
  events_match_binary = events_match
  pm_binary = np.zeros((len(pass_names),len(pass_names)))
  pm_binary[0,0] = events_match_binary.shape[0]
  pm_binary[0,1] = pass_unmatched_dcu.shape[0]
  pm_binary[1,0] = pass_unmatched_sts.shape[0]
  results['open_play']['names'] = pass_names
  results['open_play']['performance'] = pm_binary     

  dcu_shot_tags_matching = ['shot']
  dcu_pass_tags_matching = [tag for tag in dcu_pass_tags if tag not in ['cross','clearance']+dcu_shot_tags_matching]

  sts_shot_tags_matching = ['shot','goal']
  sts_pass_tags_matching = [tag for tag in sts_pass_tags if tag not in ['cross','clearance']+sts_shot_tags_matching]

  pm_open = np.zeros((5,5))
  pm_open[0,0] = events_match_binary.query("event_dcu.isin(@dcu_pass_tags_matching) and event_sts.isin(@sts_pass_tags_matching)").shape[0]
  pm_open[1,1] = events_match_binary.query("event_dcu.isin(@dcu_shot_tags_matching) and event_sts.isin(@sts_shot_tags_matching)").shape[0]  
  pm_open[2,2] = events_match_binary.query("event_dcu == 'cross' and event_sts == 'cross'").shape[0] 
  pm_open[3,3] = events_match_binary.query("event_dcu == 'clearance' and event_sts == 'clearance'").shape[0] 
  pm_open[0,1] = events_match_binary.query("event_dcu.isin(@dcu_pass_tags_matching) and event_sts.isin(@sts_shot_tags_matching)").shape[0]
  pm_open[0,2] = events_match_binary.query("event_dcu.isin(@dcu_pass_tags_matching) and event_sts == 'cross'").shape[0]
  pm_open[0,3] = events_match_binary.query("event_dcu.isin(@dcu_pass_tags_matching) and event_sts == 'clearance'").shape[0]
  pm_open[1,0] = events_match_binary.query("event_dcu.isin(@dcu_shot_tags_matching) and event_sts.isin(@sts_pass_tags_matching)").shape[0]
  pm_open[1,2] = events_match_binary.query("event_dcu.isin(@dcu_shot_tags_matching) and event_sts == 'cross'").shape[0]
  pm_open[1,2] = events_match_binary.query("event_dcu.isin(@dcu_shot_tags_matching) and event_sts == 'clearance'").shape[0]
  pm_open[2,0] = events_match_binary.query("event_dcu == 'cross' and event_sts.isin(@sts_pass_tags_matching)").shape[0]
  pm_open[2,1] = events_match_binary.query("event_dcu == 'cross' and event_sts.isin(@sts_shot_tags_matching)").shape[0]
  pm_open[2,3] = events_match_binary.query("event_dcu == 'cross' and event_sts == 'clearance'").shape[0]
  pm_open[3,0] = events_match_binary.query("event_dcu == 'clearance' and event_sts.isin(@sts_pass_tags_matching)").shape[0]
  pm_open[3,1] = events_match_binary.query("event_dcu == 'clearance' and event_sts.isin(@sts_shot_tags_matching)").shape[0]
  pm_open[3,2] = events_match_binary.query("event_dcu == 'clearance' and event_sts == 'cross'").shape[0]
  pm_open[-1,0] = pass_unmatched_sts.query("event.isin(@sts_pass_tags_matching)").shape[0]
  pm_open[-1,1] = pass_unmatched_sts.query("event.isin(@dcu_shot_tags_matching)").shape[0]
  pm_open[-1,2] = pass_unmatched_sts.query("event == 'cross'").shape[0]
  pm_open[-1,3] = pass_unmatched_sts.query("event == 'clearance'").shape[0]
  pm_open[0,-1] = pass_unmatched_dcu.query("event.isin(@dcu_pass_tags_matching)").shape[0]
  pm_open[1,-1] = pass_unmatched_dcu.query("event.isin(@dcu_shot_tags_matching)").shape[0]
  pm_open[2,-1] = pass_unmatched_dcu.query("event == 'cross'").shape[0]
  pm_open[3,-1] = pass_unmatched_dcu.query("event == 'clearance'").shape[0]
  results['pass_shot_cross']['names'] = ['pass','attempt','cross','clearance','none']
  results['pass_shot_cross']['performance'] = pm_open

  dcu_pass_tags_matching = [tag for tag in dcu_pass_tags if tag not in dcu_shot_tags_matching]
  sts_pass_tags_matching = [tag for tag in sts_pass_tags if tag not in dcu_shot_tags_matching]

  pm_open = np.zeros((3,3))
  pm_open[0,0] = events_match_binary.query("event_dcu.isin(@dcu_pass_tags_matching) and event_sts.isin(@sts_pass_tags_matching)").shape[0]
  pm_open[1,1] = events_match_binary.query("event_dcu.isin(@dcu_shot_tags_matching) and event_sts.isin(@sts_shot_tags_matching)").shape[0]  
  pm_open[0,1] = events_match_binary.query("event_dcu.isin(@dcu_pass_tags_matching) and event_sts.isin(@sts_shot_tags_matching)").shape[0]
  pm_open[1,0] = events_match_binary.query("event_dcu.isin(@dcu_shot_tags_matching) and event_sts.isin(@sts_pass_tags_matching)").shape[0]
  pm_open[-1,0] = pass_unmatched_sts.query("event.isin(@sts_pass_tags_matching)").shape[0]
  pm_open[-1,1] = pass_unmatched_sts.query("event.isin(@sts_shot_tags_matching)").shape[0]
  pm_open[0,-1] = pass_unmatched_dcu.query("event.isin(@dcu_pass_tags_matching)").shape[0]
  pm_open[1,-1] = pass_unmatched_dcu.query("event.isin(@dcu_shot_tags_matching)").shape[0]
  results['pass_shot']['names'] = ['pass/cross','attempt','none']
  results['pass_shot']['performance'] = pm_open
    
  
  ## Goals
  goal_names = ['goal','own goal','none']

  pm_goal = np.zeros((len(goal_names),len(goal_names)))
  pm_goal[0,0] = goals_match.query("event_dcu == 'goal' and event_sts == 'goal'", engine='python').shape[0]
  pm_goal[0,1] = goals_match.query("event_dcu == 'goal' and event_sts == 'own goal'", engine='python').shape[0]
  pm_goal[1,0] = goals_match.query("event_dcu == 'own goal' and event_sts == 'goal'", engine='python').shape[0]
  pm_goal[1,1] = goals_match.query("event_dcu == 'own goal' and event_sts == 'own goal'", engine='python').shape[0]
  pm_goal[0,-1] = goals_unmatched_dcu.query("event == 'goal'").shape[0]
  pm_goal[1,-1] = goals_unmatched_dcu.query("event == 'own goal'").shape[0]
  pm_goal[-1,0] = goals_unmatched_sts.query("event == 'goal'").shape[0]
  pm_goal[-1,1] = goals_unmatched_sts.query("event == 'own goal'").shape[0]

  results['goal']['names'] = goal_names
  results['goal']['performance'] = pm_goal


  ## Set pieces
  events_match_setpiece = events_match.query("wave == 'dbe'", engine='python').reset_index(drop=True)
  results_origin = dict()
  set_names = ['kickoff','penalty kick','corner kick','throw-in','goal kick','free kick','none']
  pm_set = np.zeros((len(set_names),len(set_names)))
  for i, name_i in enumerate(set_names[:-1]):
    for j, name_j in enumerate(set_names[:-1]):
        pm_set[i,j] = events_match_setpiece.loc[(events_match_setpiece['from_set_piece_dcu'] == name_i) & (events_match_setpiece['from_set_piece_sts'] == name_j)].shape[0]
  
  for j, name_j in enumerate(set_names[:-1]):
    pm_set[-1,j] = pass_unmatched_sts.query("from_set_piece == @name_j", engine='python').shape[0]
  
  for i, name_i in enumerate(set_names[:-1]):
    pm_set[i,-1] = pass_unmatched_dcu.query("from_set_piece == @name_i", engine='python').shape[0]

  results['set_piece']['names'] = set_names
  results['set_piece']['performance'] = pm_set
  results['dcu_pass_tags'] = dcu_pass_tags
  results['sts_pass_tags'] = sts_pass_tags


  return results, matching

def match_events(this_game, matching_params, provider):

    """
    This routine assigns the proper subroutines based on the events provider we want to compare to, runs the matching and returns the results


    :param this_game: Game structure containing auto-events and official STS events
    :param matching_params: Dictionary containing all the relevant parameters for matching
    :param provider: Provider to compare to
    :return: Game structured updated with results of matching
    """ 
    if provider == 'sts':
        this_game[f'events_sts'] = get_official_events_sts(this_game, matching_params)
        this_game[f'results_sts'], this_game[f'matching_sts'] = matching_algorithm_sts(this_game, matching_params)

    elif provider == 'dcu':
        this_game[f'events_dcu'] = get_official_events_dcu(this_game, matching_params)
        this_game[f'results_dcu'], this_game[f'matching_dcu'] = matching_algorithm_dcu(this_game, matching_params)

    elif provider == 'statsbomb':
        this_game[f'events_statsbomb'] = get_official_events_statsbomb(this_game, matching_params)
        this_game[f'results_statsbomb'], this_game[f'matching_statsbomb'] = matching_algorithm_statsbomb(this_game, matching_params)

    elif provider == 'opta':
        this_game[f'events_opta'] = get_official_events_opta(this_game, matching_params)
        this_game[f'results_opta'], this_game[f'matching_opta'] = matching_algorithm_opta(this_game, matching_params)

    return this_game


###########################################################################################################
###########################################################################################################
### The routines below are designed to benchmark against Statsbomb and Opta. They have not been tested with 
### raw files, only the ones already ingested in BQ (fcb for statsbomb and epl for Opta), but the adaptation
### should be straightforward
###########################################################################################################
###########################################################################################################


def get_official_events_statsbomb(params, bq_client):

    """
    This routine queries Statsbomb events from BigQuery into a dataframe for ease of matching. Need to incorporate
    duels if desired

    :param params: Dictionary containing all the relevant parameters for matching
    :param bq_client: BigQuery client for querying
    :return: Dictionary with Statsbomb passing, saving and dueling events, as well as all events and tags used for passes
    """ 

    half = int(params['half'])
    players_df = params['players_df'].reset_index().set_index('events_player_id')

    tstart = time()
    sql = f"""
            SELECT event_id, time, event_name, event_outcome,player_1,team_1, skillcorner_frame, from_set_piece, type
            FROM {params['dataset']}.events_statsbomb
            WHERE events_game_id = '{params['events_game_id']}' AND half = {half}
            ORDER by time ASC
            """
    all_ev = bq_client.query(sql).to_dataframe()

    all_ev['from_set_piece'] = all_ev.apply(lambda x: 'kickoff start' if x['time']<5000 and x['from_set_piece'] == 'kickoff' else x['from_set_piece'],axis=1)
    
    all_ev.dropna(subset=['player_1'], inplace=True)

    all_ev.insert(0,'event_time','')
    all_ev['event_time'] = all_ev['time'].apply(lambda x: GameTime.from_event_time(half, x))

    all_ev['team_1'] = all_ev['player_1'].apply(map_events_to_tracking, args=(players_df, 'events_player_id','tracking_team_id'))
    all_ev['player_1'] = all_ev['player_1'].apply(map_events_to_tracking, args=(players_df, 'events_player_id','tracking_player_id'))
    all_ev.insert(5,'jersey_team_1','')
    all_ev['jersey_team_1'] = all_ev['player_1'].apply(jersey_team, args=(params,))

    
    all_ev['event_name'] = all_ev['event_name'].apply(lambda x: x if 'own goal against' not in x else 'own goal')
    
    all_ev.rename(columns={'skillcorner_frame':'frame',
                          'time':'time_ms','type':'event_type'}, inplace=True)

    list_of_passes = ['shot','pass','cross','clearance','error','offside','miscontrol','block']
    passes_ev = all_ev.loc[all_ev['event_name'].isin(list_of_passes)].copy()
    
    all_ev['event_type'].fillna('', inplace=True)
    
    saves_ev = all_ev.query("event_name == 'goal keeper' and event_type.str.contains('|'.join(['penalty conceded','goal conceded','shot faced']))", engine='python').copy()    


    # all_ev['game_section'].fillna('', inplace=True)
    all_ev['from_set_piece'].fillna('open play', inplace=True)
    passes_ev['from_set_piece'].fillna('open play', inplace=True)
    
    
    return {'passes':passes_ev, 'all':all_ev, 'saves':saves_ev, 'pass_tags': list_of_passes}


def get_official_events_opta(params, bq_client):

    """
    This routine queries Opta events from BigQuery into a dataframe for ease of matching. Need to incorporate
    duels if desired

    :param params: Dictionary containing all the relevant parameters for matching
    :param bq_client: BigQuery client for querying
    :return: Dictionary with Statsbomb passing, saving and dueling events, as well as all events and tags used for passes
    """ 

    half = int(params['half'])
    players_df = params['players_df'].reset_index().set_index('events_player_id')
    include_oba = eq['include_oba']

    tstart = time()
    sql = f"""
            SELECT event_id, time, event_name, event_outcome,player_1,team_1,from_set_piece
            FROM {params['dataset']}.events_opta
            WHERE events_game_id = '{params['events_game_id']}' AND half = {half}
            ORDER by time ASC
            """
    all_ev = bq_client.query(sql).to_dataframe()

    all_ev.dropna(subset=['player_1'], inplace=True)
    
    all_ev.insert(0,'event_time','')
    all_ev['event_time'] = all_ev['time'].apply(lambda x: GameTime.from_event_time(half, x))

    all_ev['team_1'] = all_ev['player_1'].apply(map_events_to_tracking, args=(players_df, 'events_player_id','tracking_team_id'))
    all_ev['player_1'] = all_ev['player_1'].apply(map_events_to_tracking, args=(players_df, 'events_player_id','tracking_player_id'))
    all_ev.insert(5,'jersey_team_1','')
    all_ev['jersey_team_1'] = all_ev['player_1'].apply(jersey_team, args=(params,))

    all_ev.rename(columns={'time':'time_ms'}, inplace=True)

    list_of_passes = ['shot','pass','cross','clearance',
                      'offside pass','blocked pass','error']
    if include_oba:
        passes_ev = all_ev.loc[all_ev['event_name'].isin(list_of_passes +['ball touch'])].copy()
    else:
        passes_ev = all_ev.loc[all_ev['event_name'].isin(list_of_passes)].copy()
    
    list_of_saves = ['save','punch','claim','keeper pick-up','keeper sweeper']
    saves_ev = all_ev.query("event_name.isin(@list_of_saves)", engine='python').copy()    


    all_ev['from_set_piece'].fillna('open play', inplace=True)
    passes_ev['from_set_piece'].fillna('open play', inplace=True)    
    

    return {'passes':passes_ev, 'all':all_ev, 'saves':saves_ev, 'pass_tags':list_of_passes}




def matching_algorithm_statsbomb(this_game, matching_params):

    """
    This routine matches the auto-events in this_game['models']['events'] to the Statsbomb events. 
    Specific parts of the routine are commented for better explainability. 


    :param this_game: Game structure containing auto-events and official Statsbomb events
    :param matching_params: Dictionary containing all the relevant parameters for matching
    :return: Dictionary with the performance metrics and with actual matched events for matching passes, set pieces, saves
    """ 

    params = this_game['params']
    model = this_game['models']['events']
    official_ev = this_game['events_statsbomb'].copy()
    thres = matching_params['thresh_for_detection']
    include_oba = matching_params['include_oba']

    match_col = 'time_ms'
    factor = 1000
        
    passes_ev = official_ev['passes']
    final_model = model.query("dead_ball_event == 'goal' | event_name!=''").copy()
    final_model['from_set_piece'] = final_model['from_set_piece'].replace('','open play')
    final_model.replace({' and retain':'',' and deflect':''}, inplace=True, regex=True)
    passes_model = final_model.query("event_name.str.contains('|'.join(['pass','shot','cross','own']))", engine='python').reset_index(drop=True)
  
    pass_tags = official_ev['pass_tags']
    
    saves_ev = official_ev['saves']
    
    ############## MATCHING DEAD BALL EVENTS
    #### remove first kickoff
    dbe_model = passes_model.query("from_set_piece != 'open play' and from_set_piece !='kickoff start'").reset_index(drop=True)
    dbe_off = passes_ev.query("from_set_piece != 'open play' and from_set_piece !='kickoff start'", engine='python').reset_index(drop=True)

    first_filter = 30
    matching_dbe = []
    for row in dbe_off.itertuples():
        
        this_t = row.time_ms

        t0 = this_t - first_filter*factor
        t1 = this_t + first_filter*factor
            
        within_minute = dbe_model.loc[(dbe_model[match_col] >= t0) &
                                      (dbe_model[match_col] <= t1)].copy()
        
        if within_minute.empty:
            continue
        else:
            within_minute['same_player'] = within_minute['player_id'].apply(match_player_interval,
                                                                            args=(row.player_1,))
            within_minute['same_team'] = within_minute['team_id'].apply(match_player_interval,
                                                                        args=(row.team_1,))
            within_minute['time_delta'] = within_minute[match_col].apply(lambda x: abs(x - this_t)/factor)
                
            if row.from_set_piece == 'kickoff':
                within_minute = within_minute.loc[(within_minute['same_team'] == True) & 
                                             (within_minute['from_set_piece'].isin(['kickoff','kickoff start']))]
            else:
                within_minute = within_minute.loc[within_minute['same_player'] == True]

            if within_minute.empty:
                continue
            else:
                within_minute = within_minute.sort_values(by='time_delta').reset_index()    
                within_minute_matched_dbe = within_minute.loc[0]
                matching_dbe.append([within_minute_matched_dbe['time_delta'], 
                                     within_minute_matched_dbe['frame'],
                                     row.event_id, within_minute_matched_dbe['event_time']])
        del within_minute

    match_dbe = pd.DataFrame(matching_dbe,columns=['time_delta','frame_model','event_id','event_time_model'])

    min_deltas = match_dbe.groupby(['frame_model'])['time_delta'].transform(min)
    match_dbe = match_dbe.loc[match_dbe['time_delta'] == min_deltas].reset_index(drop=True)
    match_dbe['wave'] = 'dbe'

    ############## FIRST WAVE OF MATCHING PASSES  
    passes_ev_open = passes_ev.query("from_set_piece == 'open play'").reset_index(drop=True)
    passes_model_open = passes_model.query("from_set_piece == 'open play'", engine='python').reset_index(drop=True)

    matching_events = []
    for row in passes_ev_open.itertuples():
        
        this_t = row.time_ms
        t0 = this_t - thres*factor
        t1 = this_t + thres*factor
        
        within_minute = passes_model_open.loc[(passes_model_open[match_col] >= t0) &
                                          (passes_model_open[match_col] <= t1)].copy()
        if within_minute.empty:
            continue
        else:
            within_minute['same_player'] = within_minute['player_id'].apply(match_player_interval,args=(row.player_1,))
            within_minute['time_delta'] = within_minute[match_col].apply(lambda x: abs(x - this_t)/factor)
            within_minute = within_minute.loc[(within_minute['same_player'] == True) & (within_minute['time_delta'] < thres)]

            if within_minute.empty:
                continue
            else:
                within_minute = within_minute.sort_values(by='time_delta').reset_index()    
                within_minute_matched = within_minute.loc[0]
                matching_events.append([within_minute_matched['time_delta'], within_minute_matched['frame'],
                                    row.event_id, within_minute_matched['event_time']])
        del within_minute

    first_matching = pd.DataFrame(matching_events,columns=['time_delta','frame_model','event_id','event_time_model'])

    min_deltas = first_matching.groupby(['frame_model'])['time_delta'].transform(min)
    first_matching = first_matching.loc[first_matching['time_delta'] == min_deltas].reset_index(drop=True)
    first_matching['wave'] = 'first'


    unmatched_off = passes_ev_open.loc[~passes_ev_open['event_id'].isin(first_matching['event_id'].unique()),
                                       'event_id']
    unmatched_off = passes_ev_open[['event_time','event_name',
                                    'jersey_team_1','event_id']].merge(unmatched_off, how='right', on='event_id')

    unmatched_model = passes_model_open.loc[~passes_model_open['frame'].isin(first_matching['frame_model'].tolist()),
                                     ['event_time','event_name','jersey_team','frame','player_id','time_ms']]



    ##### JOIN DATAFRAMES ON ALL EVENTS
    all_ev = official_ev['all']
    events_match = pd.concat([first_matching, match_dbe], ignore_index=True).sort_values(by='frame_model')
    events_match = events_match.merge(all_ev[['event_time','event_name','from_set_piece','jersey_team_1',
                                            'event_id','event_outcome']], how='left', on='event_id')
    events_match = events_match.merge(passes_model[['frame','event_name',
                                                    'from_set_piece','dead_ball_event',
                                                    'step_change']].rename(columns={'frame':'frame_model'}), 
                                      how='left', on='frame_model',suffixes=('','_model'))

    finally_unmatched_model = passes_model.loc[~passes_model['frame'].isin(events_match['frame_model'].unique()),'frame']
    finally_unmatched_model = passes_model[['event_time','event_name',
                                            'from_set_piece','jersey_team','player_id',
                                            'frame','dead_ball_event',
                                            'step_change']].merge(finally_unmatched_model, how='right', on='frame')
    finally_unmatched_model['source'] = 'model'

    finally_unmatched_off = passes_ev.loc[~passes_ev['event_id'].isin(events_match['event_id'].unique()),'event_id']
    finally_unmatched_off = passes_ev[['event_time','event_name',
                                       'from_set_piece','jersey_team_1',
                                     'player_1','event_id',
                                     'event_outcome']].merge(finally_unmatched_off,
                                                                              how='right', on='event_id')

    finally_unmatched_off['source'] = 'official'
    finally_unmatched_off = finally_unmatched_off.rename(columns={'jersey_team_1':'jersey_team',
                                                                'player_1':'player_id'})

    unmatched_events = pd.concat([finally_unmatched_model,finally_unmatched_off],ignore_index=True)
    unmatched_events = unmatched_events.sort_values(by='event_time',ascending=True).reset_index(drop=True)


    ##### MATCHING FOR GOALS
    goals_ev = all_ev.query("event_outcome == 'goal' or event_name == 'own goal'").reset_index(drop=True)
    goals_model_raw = model.query("dead_ball_event == 'goal'").copy().reset_index().rename(columns={'index':'goal_index'})
    if not goals_model_raw.empty:
        ii = []
        ev = []
        for row in goals_model_raw.itertuples():
            ii.append(model.loc[:row.goal_index,'event_name'].str.contains("|".join(['shot','own']))[::-1].idxmax())
            ev.append(row.dead_ball_event)
        goals_model = model.loc[ii].copy().reset_index(drop=True)
        goals_model['dead_ball_event'] = ev
    
    else:
        goals_model = goals_model_raw.copy()
    
    goal_filter = 30
    matching_goals = []
    for row in goals_ev.itertuples():
        

        this_t = row.time_ms
        t0 = this_t - goal_filter*factor
        t1 = this_t + goal_filter*factor

        within_minute = goals_model.loc[(goals_model[match_col] >= t0) &
                                        (goals_model[match_col] <= t1)].copy()
        if within_minute.empty:
            continue
        else:
            within_minute['same_player'] = within_minute['player_id'].apply(match_player_interval,args=(row.player_1,))
            within_minute['time_delta'] = within_minute[match_col].apply(lambda x: abs(x - this_t)/factor)
            within_minute = within_minute.loc[(within_minute['same_player'] == True) & (within_minute['time_delta'] < thres)]

            if within_minute.empty:
                continue
            else:
                within_minute = within_minute.sort_values(by='time_delta').reset_index()    
                within_minute_matched = within_minute.loc[0]
                matching_goals.append([within_minute_matched['time_delta'], within_minute_matched['frame'],
                                    row.event_id, within_minute_matched['event_time']])
        del within_minute

    goals_match = pd.DataFrame(matching_goals,columns=['time_delta','frame','event_id','event_time_model'])
    goals_match = goals_match.merge(all_ev[['event_time','event_name','from_set_piece',
                                          'jersey_team_1',
                                            'event_id','event_outcome']],
                                    how='left', on='event_id')
    goals_match = goals_match.merge(goals_model[['frame','event_name',
                                                 'from_set_piece','dead_ball_event']],
                                    how='left', on='frame',suffixes=('','_model'))

    unmatched_goals_model = goals_model.loc[~goals_model['frame'].isin(goals_match['frame'].unique()),
                                            'frame']
    unmatched_goals_model = goals_model[['event_time','event_name','from_set_piece',
                                         'jersey_team',
                                       'player_id','frame','dead_ball_event']].merge(unmatched_goals_model,
                                                                                     how='right', on='frame')
    unmatched_goals_model['source'] = 'model'

    unmatched_goals_off = goals_ev.loc[~goals_ev['event_id'].isin(goals_match['event_id'].unique()),'event_id']
    unmatched_goals_off = goals_ev[['event_time','event_name','from_set_piece',
                                    'jersey_team_1',
                                     'player_1','event_id','event_outcome']].merge(unmatched_goals_off, 
                                                                                   how='right', on='event_id')

    unmatched_goals_off['source'] = 'official'
    unmatched_goals_off = unmatched_goals_off.rename(columns={'jersey_team_1':'jersey_team',
                                                                'player_1':'player_id'})
    unmatched_goals = pd.concat([unmatched_goals_model,unmatched_goals_off],ignore_index=True)
    unmatched_goals = unmatched_goals.sort_values(by='event_time',ascending=True).reset_index(drop=True)
    goals_match.drop(columns=['time_delta','event_id'],inplace=True)

    
    ##### MATCHING FOR SAVES
    saves_ev = official_ev['saves']

    # wave 1, only save/claims in official events
    saves_model = final_model.query("is_goalkeeper == True and event_name.str.contains('|'.join(['save','claim']))", 
                                    engine='python').reset_index(drop=True)
    saves_first = []
    save_filter = 20
    for row in saves_ev.itertuples():
        

        this_t = row.time_ms
        t0 = this_t - save_filter*factor
        t1 = this_t + save_filter*factor
        
        within_minute = saves_model.loc[(saves_model[match_col] >= t0) &
                                          (saves_model[match_col] <= t1)].copy()
        if within_minute.empty:
            continue
        else:
            for rr in within_minute.itertuples():
                within_minute.loc[rr.Index,'same_player'] = match_player_interval(row.player_1, rr.player_id)
            within_minute['time_delta'] = within_minute[match_col].apply(lambda x: abs(x - this_t)/factor)
            within_minute = within_minute.loc[(within_minute['same_player'] == True) & (within_minute['time_delta'] < thres)]

            if within_minute.empty:
                continue
            else:
                within_minute = within_minute.sort_values(by='time_delta').reset_index()    
                within_minute_matched = within_minute.loc[0]
                saves_first.append([within_minute_matched['time_delta'], within_minute_matched['frame'],
                                    row.event_id, within_minute_matched['event_time']])
        del within_minute

    saves_match = pd.DataFrame(saves_first,columns=['time_delta','frame','event_id','event_time_model'])
    min_deltas = saves_match.groupby(['frame'])['time_delta'].transform(min)
    saves_match = saves_match.loc[saves_match['time_delta'] == min_deltas].reset_index(drop=True)
    
    unmatched_saves_off = saves_ev.loc[~saves_ev['event_id'].isin(saves_match['event_id'].unique()),
                                       'event_id']
    unmatched_saves_off = saves_ev[['event_time','event_name',
                                    'jersey_team_1','event_id']].merge(unmatched_saves_off,
                                                                                how='right', on='event_id')
    
    unmatched_saves_off = saves_ev.loc[~saves_ev['event_id'].isin(saves_match['event_id'].unique())]
    unmatched_saves_model = saves_model.loc[~saves_model['frame'].isin(saves_match['frame'].tolist())]
    
    saves_match = saves_match.merge(saves_ev[['event_time','event_name','jersey_team_1','event_id','event_outcome']], how='left',
                                    on='event_id')
    
    saves_match = saves_match.merge(saves_model[['frame',
                                                 'event_name']], how='left',
                                    on='frame',suffixes=('','_model'))

    unmatched_saves_model = saves_model.loc[(~saves_model['frame'].isin(saves_match['frame'].unique())),
                                            'frame']
    unmatched_saves_model = saves_model[['event_time','event_name','from_set_piece',
                                        'jersey_team','player_id','frame']].merge(unmatched_saves_model,
                                                                                  how='right', on='frame')
    unmatched_saves_model['source'] = 'model'

    unmatched_saves_off = saves_ev.loc[~saves_ev['event_id'].isin(saves_match['event_id'].unique()),
                                       'event_id']
    unmatched_saves_off = saves_ev[['event_time','event_name','from_set_piece','jersey_team_1',
                                     'player_1','event_id','event_outcome']].merge(unmatched_saves_off,
                                                                                   how='right',
                                                                                   on='event_id')
    unmatched_saves_off['source'] = 'official'
    unmatched_saves_off = unmatched_saves_off.rename(columns={'jersey_team_1':'jersey_team',
                                                                            'player_1':'player_id'})

    
    unmatched_saves = pd.concat([unmatched_saves_model,unmatched_saves_off],ignore_index=True)
    unmatched_saves = unmatched_saves.sort_values(by='event_time',ascending=True).reset_index(drop=True)

    unmatched_saves.drop(columns=['frame','event_id'], inplace=True)
    saves_match.drop(columns=['time_delta','event_id','frame'], inplace=True)


    ##### CONFUSION MATRIX
    results = dict()
    results['open_play'] = dict()  
    results['pass_shot'] = dict() 
    results['pass_shot_cross'] = dict() 
    results['save'] = dict()
    results['goal'] = dict()
    results['set_piece'] = dict()

    matching = dict()
    matching['passes_match'] = events_match
    matching['passes_unmatch'] = unmatched_events
    matching['saves_match'] = saves_match
    matching['saves_unmatch'] = unmatched_saves
    matching['goals_match'] = goals_match
    matching['goals_unmatch'] = unmatched_goals

    ## PASSES/SHOTS/CROSSES
    pass_names = ['pass/shot/cross','none']
    events_match_binary = events_match.query("event_name.str.contains('|'.join(@pass_tags))", 
                                             engine='python').reset_index(drop=True)
    pm_binary = np.zeros((2,2))
    pm_binary[0,0] = events_match_binary.shape[0]
    pm_binary[0,1] = finally_unmatched_off.shape[0]
    pm_binary[1,0] = finally_unmatched_model.shape[0]
    results['open_play']['names'] = pass_names
    results['open_play']['performance'] = pm_binary
    
    shot_tags_matching = ['shot']
    pass_tags_matching = [tag for tag in pass_tags if tag not in ['cross']+shot_tags_matching]
    pm_open = np.zeros((4,4))
    pm_open[0,0] = events_match_binary.query("event_name.isin(@pass_tags_matching) and event_name_model == 'pass'").shape[0]
    pm_open[1,1] = events_match_binary.query("event_name.isin(@shot_tags_matching) and event_name_model.str.contains('shot')").shape[0]  
    pm_open[2,2] = events_match_binary.query("event_name == 'cross' and event_name_model == 'cross'").shape[0] 
    pm_open[0,1] = events_match_binary.query("event_name.isin(@pass_tags_matching) and event_name_model.str.contains('shot')").shape[0]
    pm_open[0,2] = events_match_binary.query("event_name.isin(@pass_tags_matching) and event_name_model == 'cross'").shape[0]
    pm_open[1,0] = events_match_binary.query("event_name.isin(@shot_tags_matching) and event_name_model == 'pass'").shape[0]
    pm_open[1,2] = events_match_binary.query("event_name.isin(@shot_tags_matching) and event_name_model == 'cross'").shape[0]
    pm_open[2,0] = events_match_binary.query("event_name == 'cross' and event_name_model == 'pass'").shape[0]
    pm_open[2,1] = events_match_binary.query("event_name == 'cross' and event_name_model.str.contains('shot')").shape[0]
    pm_open[-1,0] = finally_unmatched_model.query("event_name == 'pass'").shape[0]
    pm_open[-1,1] = finally_unmatched_model.query("event_name.str.contains('shot')").shape[0]
    pm_open[-1,2] = finally_unmatched_model.query("event_name == 'cross'").shape[0]
    pm_open[0,-1] = finally_unmatched_off.query("event_name.isin(@pass_tags_matching)").shape[0]
    pm_open[1,-1] = finally_unmatched_off.query("event_name.isin(@shot_tags_matching)").shape[0]
    pm_open[2,-1] = finally_unmatched_off.query("event_name == 'cross'").shape[0]
    results['pass_shot_cross']['names'] = ['pass','shot','cross','none']
    results['pass_shot_cross']['performance'] = pm_open
    
    open_names = ['pass','shot','none']
    pass_tags_matching = [tag for tag in pass_tags if tag not in shot_tags_matching]
    pm_open = np.zeros((3,3))
    pm_open[0,0] = events_match_binary.query("event_name.isin(@pass_tags_matching) and event_name_model.isin(['pass','cross'])").shape[0]
    pm_open[1,1] = events_match_binary.query("event_name.isin(@shot_tags_matching) and event_name_model.str.contains('shot')").shape[0]  
    pm_open[0,1] = events_match_binary.query("event_name.isin(@pass_tags_matching) and event_name_model.str.contains('shot')").shape[0]
    pm_open[1,0] = events_match_binary.query("event_name.isin(@shot_tags_matching) and event_name_model.isin(['pass','cross'])").shape[0]
    pm_open[-1,0] = finally_unmatched_model.query("event_name.isin(['pass','cross'])").shape[0]
    pm_open[-1,1] = finally_unmatched_model.query("event_name.str.contains('shot')").shape[0]
    pm_open[0,-1] = finally_unmatched_off.query("event_name.isin(@pass_tags_matching)").shape[0]
    pm_open[1,-1] = finally_unmatched_off.query("event_name.isin(@shot_tags_matching)").shape[0]
    results['pass_shot']['names'] = ['pass/cross','shot','none']
    results['pass_shot']['performance'] = pm_open

    
  ## SAVES
    save_names = ['save','none']
    pm_save = np.zeros((len(save_names),len(save_names)))
    if not saves_ev.empty:      
        pm_save[0,0] = saves_match.shape[0]
        pm_save[0,-1] = unmatched_saves_off.shape[0]
        pm_save[-1,0] = unmatched_saves_model.shape[0]

    results['save']['names'] = save_names
    results['save']['performance'] = pm_save
  
    ## GOALS
    goal_names = ['goal','own goal','goal?','none']

    pm_goal = np.zeros((len(goal_names),len(goal_names)))
    pm_goal[0,0] = goals_match.query("event_outcome == 'goal' and dead_ball_event == 'goal' and event_name != 'own goal' and event_name_model !='own goal'", engine='python').shape[0]
    pm_goal[0,1] = goals_match.query("event_outcome == 'goal' and dead_ball_event == 'goal' and event_name != 'own goal' and event_name_model =='own goal'", engine='python').shape[0]
    pm_goal[1,0] = goals_match.query("event_outcome == 'goal' and dead_ball_event == 'goal' and event_name == 'own goal' and event_name_model !='own goal'", engine='python').shape[0]  
    pm_goal[1,1] = goals_match.query("event_outcome == 'goal' and dead_ball_event == 'goal' and event_name == 'own goal' and event_name_model =='own goal'", engine='python').shape[0]
    pm_goal[0,2] = goals_match.query("event_outcome == 'goal' and dead_ball_event == 'goal?' and event_name != 'own goal' and event_name_model !='goal?'", engine='python').shape[0]
    pm_goal[1,2] = goals_match.query("event_outcome == 'goal' and dead_ball_event == 'goal?' and event_name == 'own goal' and event_name_model !='goal?'", engine='python').shape[0]
    pm_goal[0,-1] = unmatched_goals_off.query("event_name != 'own goal'").shape[0]
    pm_goal[1,-1] = unmatched_goals_off.query("event_name == 'own goal'").shape[0]
    pm_goal[-1,0] = unmatched_goals_model.query("dead_ball_event == 'goal' and event_name != 'own goal'").shape[0]
    pm_goal[-1,1] = unmatched_goals_model.query("dead_ball_event == 'goal' and event_name == 'own goal'").shape[0]
    pm_goal[-1,2] = unmatched_goals_model.query("dead_ball_event == 'goal?'").shape[0]

    results['goal']['names'] = goal_names
    results['goal']['performance'] = pm_goal


    ## SET PIECE
    events_match_setpiece = events_match.query("event_name.str.contains('|'.join(@pass_tags))", engine='python').reset_index(drop=True)
    results_origin = dict()
    set_names = ['open play','kickoff','penalty kick','corner kick','throw-in','goal kick',
                 'free kick','free kick?','none']
    pm_set = np.zeros((len(set_names),len(set_names)))
    for i, name_i in enumerate(set_names[:-1]):
        for j, name_j in enumerate(set_names[:-1]):
            pm_set[i,j] = events_match_setpiece.loc[(events_match_setpiece['from_set_piece'] == name_i) & (events_match_setpiece['from_set_piece_model'] == name_j)].shape[0]

    for j, name_j in enumerate(set_names[:-1]):
        pm_set[-1,j] = finally_unmatched_model.query("from_set_piece == @name_j and event_name.str.contains('|'.join(['pass','shot','cross']))", engine='python').shape[0]

    for i, name_i in enumerate(set_names[:-1]):
        pm_set[i,-1] = finally_unmatched_off.query("from_set_piece == @name_i and event_name.str.contains('|'.join(@pass_tags))", engine='python').shape[0]
    
    set_names[-2] = '?'
    results['set_piece']['names'] = set_names
    results['set_piece']['performance'] = pm_set
    results['pass_tags'] = pass_tags
    
    return results, matching


def matching_algorithm_opta(this_game, matching_params):

    """
    This routine matches the auto-events in this_game['models']['events'] to the STS events. 
    Specific parts of the routine are commented for better explainability. 


    :param this_game: Game structure containing auto-events and official STS events
    :param matching_params: Dictionary containing all the relevant parameters for matching
    :return: Dictionary with the performance metrics and with actual matched events for matching passes, set pieces, saves, duels
    """ 

    params = this_game['params']
    model = this_game['models']['events']
    model['frame'] = model['frame'].astype(str)    
    official_ev = this_game['events_opta'].copy()
    thres = matching_params['thresh_for_detection']
    include_oba = matching_params['include_oba']
    
    match_col = 'time_ms'
    factor = 1000

    passes_ev = official_ev['passes']
    final_model = model.query("dead_ball_event == 'goal' | event_name!=''").copy()
    final_model['from_set_piece'] = final_model['from_set_piece'].replace('','open play')
    final_model.replace({' and retain':'',' and deflect':''}, inplace=True, regex=True)
    passes_model = final_model.query("event_name.str.contains('|'.join(['pass','shot','cross']))", engine='python').reset_index(drop=True)
  
    pass_tags = official_ev['pass_tags']
    
    saves_ev = official_ev['saves']
    
    ############## MATCHING DEAD BALL EVENTS
    #### remove first kickoff
    dbe_model = passes_model.query("from_set_piece != 'open play' and from_set_piece !='kickoff start'").reset_index(drop=True)
    dbe_off = passes_ev.query("from_set_piece != 'open play' and from_set_piece !='kickoff start'", engine='python').reset_index(drop=True)

    first_filter = 30
    matching_dbe = []
    for row in dbe_off.itertuples():

        t0 = row.time_ms - first_filter*factor
        t1 = row.time_ms + first_filter*factor
            
        within_minute = dbe_model.loc[(dbe_model[match_col] >= t0) &
                                      (dbe_model[match_col] <= t1)].copy()
        
        if within_minute.empty:
            continue
        else:
            within_minute['same_player'] = within_minute['player_id'].apply(match_player_interval,
                                                                            args=(row.player_1,))
            within_minute['same_team'] = within_minute['team_id'].apply(match_player_interval,
                                                                        args=(row.team_1,))
            within_minute['time_delta'] = within_minute[match_col].apply(lambda x: abs(x - row.time_ms)/factor)
                
            if row.from_set_piece == 'kickoff':
                within_minute = within_minute.loc[(within_minute['same_team'] == True) & 
                                             (within_minute['from_set_piece'].isin(['kickoff','kickoff start']))]
            else:
                within_minute = within_minute.loc[within_minute['same_player'] == True]

            if within_minute.empty:
                continue
            else:
                within_minute = within_minute.sort_values(by='time_delta').reset_index()    
                within_minute_matched_dbe = within_minute.loc[0]
                matching_dbe.append([within_minute_matched_dbe['time_delta'], 
                                     within_minute_matched_dbe['frame'],
                                     row.event_id, within_minute_matched_dbe['event_time']])
        del within_minute

    match_dbe = pd.DataFrame(matching_dbe,columns=['time_delta','frame_model','event_id','event_time_model'])

    min_deltas = match_dbe.groupby(['frame_model'])['time_delta'].transform(min)
    match_dbe = match_dbe.loc[match_dbe['time_delta'] == min_deltas].reset_index(drop=True)
    match_dbe['wave'] = 'dbe'

    ############## FIRST WAVE OF MATCHING PASSES
  
    passes_ev_open = passes_ev.query("from_set_piece == 'open play'").reset_index(drop=True)
    passes_model_open = passes_model.query("from_set_piece == 'open play'", engine='python').reset_index(drop=True)

    matching_events = []
    for row in passes_ev_open.itertuples():

        t0 = row.time_ms - thres*factor
        t1 = row.time_ms + thres*factor
        
        within_minute = passes_model_open.loc[(passes_model_open[match_col] >= t0) &
                                          (passes_model_open[match_col] <= t1)].copy()
        if within_minute.empty:
            continue
        else:
            within_minute['same_player'] = within_minute['player_id'].apply(match_player_interval,args=(row.player_1,))
            within_minute['time_delta'] = within_minute[match_col].apply(lambda x: abs(x - row.time_ms)/factor)
            within_minute = within_minute.loc[(within_minute['same_player'] == True) & (within_minute['time_delta'] < thres)]

            if within_minute.empty:
                continue
            else:
                within_minute = within_minute.sort_values(by='time_delta').reset_index()    
                within_minute_matched = within_minute.loc[0]
                matching_events.append([within_minute_matched['time_delta'], within_minute_matched['frame'],
                                    row.event_id, within_minute_matched['event_time']])
        del within_minute

    first_matching = pd.DataFrame(matching_events,columns=['time_delta','frame_model','event_id','event_time_model'])

    min_deltas = first_matching.groupby(['frame_model'])['time_delta'].transform(min)
    first_matching = first_matching.loc[first_matching['time_delta'] == min_deltas].reset_index(drop=True)
    first_matching['wave'] = 'first'


    unmatched_off = passes_ev_open.loc[~passes_ev_open['event_id'].isin(first_matching['event_id'].unique()),
                                       'event_id']
    unmatched_off = passes_ev_open[['event_time','event_name',
                                    'jersey_team_1','event_id']].merge(unmatched_off, how='right', on='event_id')

    unmatched_model = passes_model_open.loc[~passes_model_open['frame'].isin(first_matching['frame_model'].tolist()),
                                     ['event_time','event_name','jersey_team','frame','player_id','time_ms']]


  

    ##### JOIN DATAFRAMES ON ALL EVENTS
    all_ev = official_ev['all']
    events_match = pd.concat([first_matching, match_dbe], ignore_index=True).sort_values(by='frame_model')
    events_match = events_match.merge(all_ev[['event_time','event_name','from_set_piece','jersey_team_1',
                                            'event_id','event_outcome']], how='left', on='event_id')
    events_match = events_match.merge(passes_model[['frame','event_name',
                                                    'from_set_piece','dead_ball_event',
                                                    'step_change']].rename(columns={'frame':'frame_model'}), 
                                      how='left', on='frame_model',suffixes=('','_model'))

    finally_unmatched_model = passes_model.loc[~passes_model['frame'].isin(events_match['frame_model'].unique()),'frame']
    finally_unmatched_model = passes_model[['event_time','event_name',
                                            'from_set_piece','jersey_team','player_id',
                                            'frame','dead_ball_event',
                                            'step_change']].merge(finally_unmatched_model, how='right', on='frame')
    finally_unmatched_model['source'] = 'model'

    finally_unmatched_off = passes_ev.loc[~passes_ev['event_id'].isin(events_match['event_id'].unique()),'event_id']
    finally_unmatched_off = passes_ev[['event_time','event_name',
                                       'from_set_piece','jersey_team_1',
                                     'player_1','event_id',
                                     'event_outcome']].merge(finally_unmatched_off,
                                                                              how='right', on='event_id')

    finally_unmatched_off['source'] = 'official'
    finally_unmatched_off = finally_unmatched_off.rename(columns={'jersey_team_1':'jersey_team',
                                                                'player_1':'player_id'})

    unmatched_events = pd.concat([finally_unmatched_model,finally_unmatched_off],ignore_index=True)
    unmatched_events = unmatched_events.sort_values(by='event_time',ascending=True).reset_index(drop=True)


    ##### MATCHING FOR GOALS
    goals_ev = all_ev.query("event_outcome == 'goal' or event_name == 'own goal'").reset_index(drop=True)
    goals_model_raw = model.query("dead_ball_event == 'goal'").copy().reset_index().rename(columns={'index':'goal_index'})
    if not goals_model_raw.empty:
        ii = []
        ev = []
        for row in goals_model_raw.itertuples():
            ii.append(model.loc[:row.goal_index,'event_name'].str.contains("|".join(['shot','own']))[::-1].idxmax())
            ev.append(row.dead_ball_event)
        goals_model = model.loc[ii].copy().reset_index(drop=True)
        goals_model['dead_ball_event'] = ev
    
    else:
        goals_model = goals_model_raw.copy()
    
    goal_filter = 30
    matching_goals = []
    for row in goals_ev.itertuples():

        t0 = row.time_ms - goal_filter*factor
        t1 = row.time_ms + goal_filter*factor

        within_minute = goals_model.loc[(goals_model[match_col] >= t0) &
                                        (goals_model[match_col] <= t1)].copy()
        if within_minute.empty:
            continue
        else:
            within_minute['same_player'] = within_minute['player_id'].apply(match_player_interval,args=(row.player_1,))
            within_minute['time_delta'] = within_minute[match_col].apply(lambda x: abs(x - row.time_ms)/factor)
            within_minute = within_minute.loc[(within_minute['same_player'] == True) & (within_minute['time_delta'] < thres)]

            if within_minute.empty:
                continue
            else:
                within_minute = within_minute.sort_values(by='time_delta').reset_index()    
                within_minute_matched = within_minute.loc[0]
                matching_goals.append([within_minute_matched['time_delta'], within_minute_matched['frame'],
                                    row.event_id, within_minute_matched['event_time']])
        del within_minute

    goals_match = pd.DataFrame(matching_goals,columns=['time_delta','frame','event_id','event_time_model'])
    goals_match = goals_match.merge(all_ev[['event_time','event_name','from_set_piece',
                                          'jersey_team_1',
                                            'event_id','event_outcome']],
                                    how='left', on='event_id')
    goals_match = goals_match.merge(goals_model[['frame','event_name',
                                                 'from_set_piece','dead_ball_event']],
                                    how='left', on='frame',suffixes=('','_model'))

    unmatched_goals_model = goals_model.loc[~goals_model['frame'].isin(goals_match['frame'].unique()),
                                            'frame']
    unmatched_goals_model = goals_model[['event_time','event_name','from_set_piece',
                                         'jersey_team',
                                       'player_id','frame','dead_ball_event']].merge(unmatched_goals_model,
                                                                                     how='right', on='frame')
    unmatched_goals_model['source'] = 'model'

    unmatched_goals_off = goals_ev.loc[~goals_ev['event_id'].isin(goals_match['event_id'].unique()),'event_id']
    unmatched_goals_off = goals_ev[['event_time','event_name','from_set_piece',
                                    'jersey_team_1',
                                     'player_1','event_id','event_outcome']].merge(unmatched_goals_off, 
                                                                                   how='right', on='event_id')

    unmatched_goals_off['source'] = 'official'
    unmatched_goals_off = unmatched_goals_off.rename(columns={'jersey_team_1':'jersey_team',
                                                                'player_1':'player_id'})
    unmatched_goals = pd.concat([unmatched_goals_model,unmatched_goals_off],ignore_index=True)
    unmatched_goals = unmatched_goals.sort_values(by='event_time',ascending=True).reset_index(drop=True)
    goals_match.drop(columns=['time_delta','event_id'],inplace=True)

    
    ##### MATCHING FOR SAVES
    saves_ev = official_ev['saves']

    # wave 1, only save/claims in official events
    saves_model = final_model.query("is_goalkeeper == True and event_name.str.contains('|'.join(['save','claim','loose ball']))", engine='python').reset_index(drop=True)
    saves_first = []
    save_filter = 20
    for row in saves_ev.itertuples():

        t0 = row.time_ms - save_filter*factor
        t1 = row.time_ms + save_filter*factor
        
        within_minute = saves_model.loc[(saves_model[match_col] >= t0) &
                                          (saves_model[match_col] <= t1)].copy()
        if within_minute.empty:
            continue
        else:
            for rr in within_minute.itertuples():
                within_minute.loc[rr.Index,'same_player'] = match_player_interval(row.player_1, rr.player_id)
            within_minute['time_delta'] = within_minute[match_col].apply(lambda x: abs(x - row.time_ms)/factor)
            within_minute = within_minute.loc[(within_minute['same_player'] == True) & (within_minute['time_delta'] < thres)]

            if within_minute.empty:
                continue
            else:
                within_minute = within_minute.sort_values(by='time_delta').reset_index()    
                within_minute_matched = within_minute.loc[0]
                saves_first.append([within_minute_matched['time_delta'], within_minute_matched['frame'],
                                    row.event_id, within_minute_matched['event_time']])
        del within_minute

    saves_match = pd.DataFrame(saves_first,columns=['time_delta','frame','event_id','event_time_model'])
    min_deltas = saves_match.groupby(['frame'])['time_delta'].transform(min)
    saves_match = saves_match.loc[saves_match['time_delta'] == min_deltas].reset_index(drop=True)
    
    unmatched_saves_off = saves_ev.loc[~saves_ev['event_id'].isin(saves_match['event_id'].unique()),
                                       'event_id']
    unmatched_saves_off = saves_ev[['event_time','event_name',
                                    'jersey_team_1','event_id']].merge(unmatched_saves_off,
                                                                                how='right', on='event_id')
    
    unmatched_saves_off = saves_ev.loc[~saves_ev['event_id'].isin(saves_match['event_id'].unique())]
    unmatched_saves_model = saves_model.loc[~saves_model['frame'].isin(saves_match['frame'].tolist())]
    
    saves_match = saves_match.merge(saves_ev[['event_time','event_name','jersey_team_1','event_id','event_outcome']], how='left',
                                    on='event_id')
    
    saves_match = saves_match.merge(saves_model[['frame',
                                                 'event_name']], how='left',
                                    on='frame',suffixes=('','_model'))

    unmatched_saves_model = saves_model.loc[(~saves_model['frame'].isin(saves_match['frame'].unique())),
                                            'frame']
    unmatched_saves_model = saves_model[['event_time','event_name','from_set_piece',
                                        'jersey_team','player_id','frame']].merge(unmatched_saves_model,
                                                                                  how='right', on='frame')
    unmatched_saves_model['source'] = 'model'

    unmatched_saves_off = saves_ev.loc[~saves_ev['event_id'].isin(saves_match['event_id'].unique()),
                                       'event_id']
    unmatched_saves_off = saves_ev[['event_time','event_name','from_set_piece','jersey_team_1',
                                     'player_1','event_id','event_outcome']].merge(unmatched_saves_off,
                                                                                   how='right',
                                                                                   on='event_id')
    unmatched_saves_off['source'] = 'official'
    unmatched_saves_off = unmatched_saves_off.rename(columns={'jersey_team_1':'jersey_team',
                                                              'player_1':'player_id'})

    
    unmatched_saves = pd.concat([unmatched_saves_model,unmatched_saves_off],ignore_index=True)
    unmatched_saves = unmatched_saves.sort_values(by='event_time',ascending=True).reset_index(drop=True)

    unmatched_saves.drop(columns=['frame','event_id'], inplace=True)
    saves_match.drop(columns=['time_delta','event_id','frame'], inplace=True)


    ##### CONFUSION MATRIX
    results = dict()
    results['open_play'] = dict()  
    results['pass_shot'] = dict() 
    results['pass_shot_cross'] = dict() 
    results['save'] = dict()
    results['goal'] = dict()
    results['set_piece'] = dict()

    matching = dict()
    matching['passes_match'] = events_match
    matching['passes_unmatch'] = unmatched_events
    matching['saves_match'] = saves_match
    matching['saves_unmatch'] = unmatched_saves
    matching['goals_match'] = goals_match
    matching['goals_unmatch'] = unmatched_goals

    ## PASSES/SHOTS
    pass_names = ['pass/shot/cross','none']
    if include_oba:        
        pass_tags += 'ball touch'
    events_match_binary = events_match.query("event_name.str.contains('|'.join(@pass_tags))", 
                                             engine='python').reset_index(drop=True)
    pm_binary = np.zeros((len(pass_names),len(pass_names)))
    pm_binary[0,0] = events_match_binary.shape[0]
    pm_binary[0,1] = finally_unmatched_off.shape[0]
    pm_binary[1,0] = finally_unmatched_model.shape[0]
    results['open_play']['names'] = pass_names
    results['open_play']['performance'] = pm_binary
    
    shot_tags_matching = ['shot']
    pass_tags_matching = [tag for tag in pass_tags if tag not in ['cross']+shot_tags_matching]
    pm_open = np.zeros((4,4))
    pm_open[0,0] = events_match_binary.query("event_name.isin(@pass_tags_matching) and event_name_model == 'pass'").shape[0]
    pm_open[1,1] = events_match_binary.query("event_name.isin(@shot_tags_matching) and event_name_model.str.contains('shot')").shape[0]  
    pm_open[2,2] = events_match_binary.query("event_name == 'cross' and event_name_model == 'cross'").shape[0] 
    pm_open[0,1] = events_match_binary.query("event_name.isin(@pass_tags_matching) and event_name_model.str.contains('shot')").shape[0]
    pm_open[0,2] = events_match_binary.query("event_name.isin(@pass_tags_matching) and event_name_model == 'cross'").shape[0]
    pm_open[1,0] = events_match_binary.query("event_name.isin(@shot_tags_matching) and event_name_model == 'pass'").shape[0]
    pm_open[1,2] = events_match_binary.query("event_name.isin(@shot_tags_matching) and event_name_model == 'cross'").shape[0]
    pm_open[2,0] = events_match_binary.query("event_name == 'cross' and event_name_model == 'pass'").shape[0]
    pm_open[2,1] = events_match_binary.query("event_name == 'cross' and event_name_model == 'shot'").shape[0]
    pm_open[-1,0] = finally_unmatched_model.query("event_name == 'pass'").shape[0]
    pm_open[-1,1] = finally_unmatched_model.query("event_name.str.contains('shot')").shape[0]
    pm_open[-1,2] = finally_unmatched_model.query("event_name == 'cross'").shape[0]
    pm_open[0,-1] = finally_unmatched_off.query("event_name.isin(@pass_tags_matching)").shape[0]
    pm_open[1,-1] = finally_unmatched_off.query("event_name.isin(@shot_tags_matching)").shape[0]
    pm_open[2,-1] = finally_unmatched_off.query("event_name == 'cross'").shape[0]
    results['pass_shot_cross']['names'] = ['pass','shot','cross','none']
    results['pass_shot_cross']['performance'] = pm_open
    
    open_names = ['pass','shot','none']
    pass_tags_matching = [tag for tag in pass_tags if tag not in shot_tags_matching]
    pm_open = np.zeros((3,3))
    pm_open[0,0] = events_match_binary.query("event_name.isin(@pass_tags_matching) and event_name_model.isin(['pass','cross'])").shape[0]
    pm_open[1,1] = events_match_binary.query("event_name.isin(@shot_tags_matching) and event_name_model.str.contains('shot')").shape[0]  
    pm_open[0,1] = events_match_binary.query("event_name.isin(@pass_tags_matching) and event_name_model.str.contains('shot')").shape[0]
    pm_open[1,0] = events_match_binary.query("event_name.isin(@shot_tags_matching) and event_name_model.isin(['pass','cross'])").shape[0]
    pm_open[-1,0] = finally_unmatched_model.query("event_name.isin(['pass','cross'])").shape[0]
    pm_open[-1,1] = finally_unmatched_model.query("event_name.str.contains('shot')").shape[0]
    pm_open[0,-1] = finally_unmatched_off.query("event_name.isin(@pass_tags_matching)").shape[0]
    pm_open[1,-1] = finally_unmatched_off.query("event_name.isin(@shot_tags_matching)").shape[0]
    results['pass_shot']['names'] = ['pass/cross','shot','none']
    results['pass_shot']['performance'] = pm_open
    
    ## SAVES
    save_names = ['save','claim','reception','none']
    pm_save = np.zeros((len(save_names),len(save_names)))
    if not saves_ev.empty:      
        pm_save[0,0] = saves_match.query("event_name.isin(['save','punch']) and event_name_model == 'save'", engine='python').shape[0]
        pm_save[0,1] = saves_match.query("event_name.isin(['save','punch']) and event_name_model == 'claim'", engine='python').shape[0]
        pm_save[0,2] = saves_match.query("event_name.isin(['save','punch']) and event_name_model.str.contains('loose')", engine='python').shape[0]
        pm_save[0,-1] = unmatched_saves_off.query("event_name.isin(['save','punch'])").shape[0]

        pm_save[1,0] = saves_match.query("event_name.isin(['claim','keeper sweeper']) and event_name_model == 'save'", engine='python').shape[0]
        pm_save[1,1] = saves_match.query("event_name.isin(['claim','keeper sweeper']) and event_name_model == 'claim'", engine='python').shape[0]
        pm_save[1,2] = saves_match.query("event_name.isin(['claim','keeper sweeper']) and event_name_model.str.contains('loose')",
                                         engine='python').shape[0]
        pm_save[1,-1] = unmatched_saves_off.query("event_name.isin(['claim','keeper sweeper'])", engine='python').shape[0]

        pm_save[2,0] = saves_match.query("event_name == 'keeper pick-up' and event_name_model == 'save'", 
                                         engine='python').shape[0]
        pm_save[2,1] = saves_match.query("event_name == 'keeper pick-up' and event_name_model == 'claim'", 
                                         engine='python').shape[0]
        pm_save[2,2] = saves_match.query("event_name == 'keeper pick-up' and event_name_model.str.contains('loose')",
                                         engine='python').shape[0]
        pm_save[2,-1] = unmatched_saves_off.query("event_name == 'keeper pick-up'", engine='python').shape[0]

        pm_save[-1,0] = unmatched_saves_model.query("event_name == 'save'", engine='python').shape[0]
        pm_save[-1,1] = unmatched_saves_model.query("event_name == 'claim'", engine='python').shape[0]
        pm_save[-1,2] = unmatched_saves_model.query("event_name.str.contains('loose')", engine='python').shape[0]

    results['save']['names'] = save_names
    results['save']['performance'] = pm_save
  
    ## GOALS
    goal_names = ['goal','own goal','goal?','none']

    pm_goal = np.zeros((len(goal_names),len(goal_names)))
    pm_goal[0,0] = goals_match.query("event_outcome == 'goal' and dead_ball_event == 'goal' and event_name != 'own goal' and event_name_model !='own goal'", engine='python').shape[0]
    pm_goal[0,1] = goals_match.query("event_outcome == 'goal' and dead_ball_event == 'goal' and event_name != 'own goal' and event_name_model =='own goal'", engine='python').shape[0]
    pm_goal[1,0] = goals_match.query("event_outcome == 'goal' and dead_ball_event == 'goal' and event_name == 'own goal' and event_name_model !='own goal'", engine='python').shape[0]  
    pm_goal[1,1] = goals_match.query("event_outcome == 'goal' and dead_ball_event == 'goal' and event_name == 'own goal' and event_name_model =='own goal'", engine='python').shape[0]
    pm_goal[0,2] = goals_match.query("event_outcome == 'goal' and dead_ball_event == 'goal?' and event_name != 'own goal' and event_name_model !='goal?'", engine='python').shape[0]
    pm_goal[1,2] = goals_match.query("event_outcome == 'goal' and dead_ball_event == 'goal?' and event_name == 'own goal' and event_name_model !='goal?'", engine='python').shape[0]
    pm_goal[0,-1] = unmatched_goals_off.query("event_name != 'own goal'").shape[0]
    pm_goal[1,-1] = unmatched_goals_off.query("event_name == 'own goal'").shape[0]
    pm_goal[-1,0] = unmatched_goals_model.query("dead_ball_event == 'goal' and event_name != 'own goal'").shape[0]
    pm_goal[-1,1] = unmatched_goals_model.query("dead_ball_event == 'goal' and event_name == 'own goal'").shape[0]
    pm_goal[-1,2] = unmatched_goals_model.query("dead_ball_event == 'goal?'").shape[0]

    results['goal']['names'] = goal_names
    results['goal']['performance'] = pm_goal


    ## SET PIECE
    events_match_setpiece = events_match.query("event_name.str.contains('|'.join(@pass_tags))", engine='python').reset_index(drop=True)
    results_origin = dict()
    set_names = ['open play','kickoff','penalty kick','corner kick','throw-in','goal kick',
                 'free kick','free kick?','none']
    pm_set = np.zeros((len(set_names),len(set_names)))
    for i, name_i in enumerate(set_names[:-1]):
        for j, name_j in enumerate(set_names[:-1]):
            pm_set[i,j] = events_match_setpiece.loc[(events_match_setpiece['from_set_piece'] == name_i) & (events_match_setpiece['from_set_piece_model'] == name_j)].shape[0]

    for j, name_j in enumerate(set_names[:-1]):
        pm_set[-1,j] = finally_unmatched_model.query("from_set_piece == @name_j and event_name.str.contains('|'.join(['pass','shot','cross']))", engine='python').shape[0]

    for i, name_i in enumerate(set_names[:-1]):
        pm_set[i,-1] = finally_unmatched_off.query("from_set_piece == @name_i and event_name.str.contains('|'.join(@pass_tags))", engine='python').shape[0]
    
    results['set_piece']['names'] = set_names
    results['set_piece']['performance'] = pm_set
    results['pass_tags'] = pass_tags
    
    return results, matching

