import pandas as pd
import numpy as np



def get_params(provider, dataset, **kwargs):
    """
    This routine sets the default parameters for the algorithm

    :param provider: Tracking data provider
    :param dataset: Dataset where the data is located
    :param kwargs: Keyword parameters supplied by user to overwrite default params
    :return: Dictionary with params to run the algorithm
    """                  
    params = dict()

    ## General parameters
    params['provider'] = provider
    params['dataset'] = dataset
    params['fps'] = 25
    params['verbose'] = True
    params['set_event_logic'] = True
    params['set_possession_logic'] = True
    params['postprocess'] = False
    params['compute_modes'] = False
    params['compute_pressures'] = False
    params['compute_pitch_control'] = False

    params['pitch_measures'] = {'penspot_dim':10.9728,
                                'smbx_dim':5.4864,
                                'goal_dim':7.3152,
                                'arc_dim':9.144,
                                'penbox_len':16.4592,
                                'penbox_wid':40.2336,
                                'goal_height':2.44
                               }
    params['active_goal'] = dict()

    ## Game parameters
    params['game_id'] = ''
    params['half'] = 0
    params['segment_id'] = ''
    params['td'] = pd.DataFrame()
    params['td_frame'] = {}
    params['ball_data'] = pd.DataFrame()
    params['home_id'] = ''
    params['away_id'] = ''
    params['home_team_name'] = ''
    params['away_team_name'] = ''
    params['home_goalkeeper_ids'] = []
    params['away_goalkeeper_ids'] = []
    params['goalkeeper_ids'] = []
    params['players_df'] = pd.DataFrame()
    params['pitch_length'] = 105
    params['pitch_width'] = 68
    params['max_halves'] = 4


    ## Tolerances
    if provider == 'skillcorner':
        params['pz_setpiece'] = 5
    elif provider == '2s':
        params['pz_setpiece'] = 3
    else:
        params['pz_setpiece'] = 3
    params['player_in_pitch'] ={'sideline':1, 'endline':0.5, 'goal':0.1}
    params['ball_in_pitch'] = 1
    params['in_area'] = 0.25
    params['kickoff_player_out_pitch'] = 1
    params['on_target_threshold'] = 0.25
    params['save_deflect_interval'] = 1
    params['cross_distance'] = 10
    params['cross_height'] = 0.6
    params['goal_var'] = 5
    params['tol_ck'] = 3
    params['tol_gk'] = 0.5
    params['tol_ti'] = 0.5
    params['tol_pk1'] = 2
    params['tol_pk2'] = 2
    params['tol_pk3'] = 6
    params['tol_ko1'] =  5
    params['tol_ko2'] =  3
    params['tol_z'] = 1
    params['set_piece_max_displacement'] = 1
    params['dead_interval_ball_max_displacement'] = 5
    params['kickoff_ball_max_displacement'] = 5
    params['aerial_z'] = 1.5
    params['gk_z_claim'] = 2
    params['event_logic_lookbehind'] = 5
    
    ## Tracking data filtering
    params['filter_ball_speed'] = (2, 7)
    params['filter_ball_position'] = (2, 0)
    params['filter_player_speed'] = (2, 7)
                      

    params['db_from_sp'] = {"throw-in":"out for throw-in",
                           "goal kick":"out for goal kick",
                           "corner kick":"out for corner kick",
                           "free kick":"foul",
                           "kickoff":"goal",
                            "kickoff start":"game start",
                           "penalty kick":"penalty awarded",
                            "free kick?":"foul?"}

    ## Team possession
    params['possession'] = {'time_thresh':1,
                                   'pass_thresh':2,
                                   'set_piece_thresh':10}

    ## Pressures
    params['pressure'] = {'gk_z_limit':(1,2.5),
                       'angle_towards':60,
                       'moving_closer':0.5,
                        'radius':7,
                       'radius_back':2,
                        'min_speed':3.5,
                      'gk_possession_lookahead':1,
                         'max_team_press_cutoff':5,
                  'gk_z_threshold':0.5,
                  'infielder_z_threshold':1,
                  'high_ball_z':3,
                  'high_ball_duration':1}

    
    ## Pitch control
    params['pitch_control'] = {'vision_angle':30,
                              'pc_granularity':1}

    ## Defensive modes                        
    params['defensive_modes'] =  {
                            'high_press_min_advanced_defenders': 6,
                            'mid_block_min_defenders_opp_half': 4,
                            'mid_block_pitch_fraction': 1/3,
                            'low_block_max_defenders_opp_half': 3,
                            'min_interval_duration': 3,
                                 'transition_period_duration': 3}

    

    params.update(kwargs)
    if params['postprocess'] and not params['set_possession_logic']:
        params['set_possession_logic'] = True
        
    ## Hyperparameters
    hyperparams = dict()
    hyperparams['ball_change_direction'] = np.cos(10*np.pi/180)
    hyperparams['ball_change_speed'] = 10
    hyperparams['min_ball_displacement'] = 0.1
    hyperparams['continued_possession'] = 0

    if provider == 'skillcorner':
        hyperparams['pz'] = 2.5
        hyperparams['pz_gk'] = 2.5 
    elif provider == '2s':
        hyperparams['pz'] = 1.5
        hyperparams['pz_gk'] = 1.5   
    else:
        hyperparams['pz'] = 1
        hyperparams['pz_gk'] = 1

    hyperparams['pz_contest'] = max(1.5, hyperparams['pz'])    
    if 'hyperparams' in kwargs:
        hyperparams.update(kwargs['hyperparams'])
    params['hyperparams'] = hyperparams

    return params


def load_game_params(game_id, players_df):
    """
    This routine sets the game parameters from the players_df dataframe

    :param game_id: String for game_id
    :param players_df: Dataframe of players, one player per row
    :return: Dictionary with game params
    """  

    game_params = {}
    game_params['game_id'] = game_id  
    game_params['home_id'] = str(players_df.iloc[0]['tracking_home_id'])
    game_params['away_id'] = str(players_df.iloc[0]['tracking_away_id'])
    game_params['home_team_name'] = players_df.iloc[0]['home_team_name']
    game_params['away_team_name'] = players_df.iloc[0]['away_team_name']    
    game_params['home_acronym'] = players_df.iloc[0]['home_acronym'] if 'home_acronym' in players_df.columns else players_df.iloc[0]['home_team_name']
    game_params['away_acronym'] = players_df.iloc[0]['away_acronym'] if 'away_acronym' in players_df.columns else players_df.iloc[0]['away_team_name']        
    game_params['home_goalkeeper_ids'] = players_df.query("is_goalkeeper == 1 and is_home == 1")['tracking_player_id'].tolist()
    game_params['away_goalkeeper_ids'] = players_df.query("is_goalkeeper == 1 and is_home == 0")['tracking_player_id'].tolist()
    game_params['goalkeeper_ids'] = game_params['home_goalkeeper_ids'] + game_params['away_goalkeeper_ids']
    
    players_dict = {
            'home': {}
            , 'away': {}
        }
    for p in players_df.itertuples(index=False):
        players_dict['home' if p.is_home else 'away'][str(p.jersey_number)] = p._asdict()

    game_params['players_dict'] = players_dict
    game_params['players_df'] = players_df.set_index('tracking_player_id', drop=False)
    

    return game_params

def parse_lineups(metadata_path, game_id=None):
    """
    This routine parses the IFES metadata for players, teams and matches for a given game (if supplied) or
    for the entire tournament

    :param metadata_path: Path to the IFES metadata
    :param game_id: Selected game
    :return: Dictionary of dataframes (one per game) or single dataframe if game_id is supplied
    """  

    df_players = pd.read_excel(metadata_path+'Player IDs.xlsx').dropna(how='all')
    df_players.rename(columns={'TeamName':'team_name','PlayerShirtNumber':'jersey_number',
                               'PlayerPosition':'position','TeamDelegateDisplayName':'name',
                               'TeamDelegateIfesId':'player_id',
                               'TeamDelegateDateofBirth':'dob',
                               'TeamIfesId':'team_id',
                               'TeamDelegateFifaNationalityCountryCode':'team_acronym',
                               'CompetitionId':'tournament_id'},inplace=True)
    df_players = df_players.query("Replacements!='OUT'")

    df_players = df_players.astype({'player_id':'str','team_id':'str',
                                    'jersey_number':'str','dob':'str','tournament_id':'str'}, errors='raise')
    df_players = df_players[['player_id','name','jersey_number','position','dob','team_id',
                             'team_name','tournament_id']]

    
    df_teams = pd.read_excel(metadata_path+'Team IDs.xlsx')
    df_teams.rename(columns={'Organisation ID':'team_id','Team':'team_name'},inplace=True)
    df_teams = df_teams.astype({'team_id':'str'}, errors='raise')
    df_teams['team_acronym'] = df_teams['team_name'].apply(lambda x: x.split(' ')[-1].replace('(','').replace(')',''))
    df_teams['team_name'] = df_teams['team_name'].apply(lambda x: ' '.join(x.split(' ')[:-1]))
    df_players = df_players.merge(df_teams[['team_id','team_acronym']], on='team_id', how='left')
    map_acronym = df_players[['team_name','team_acronym']].drop_duplicates().set_index('team_name')['team_acronym'].to_dict()


    df_matches = pd.read_excel(metadata_path+'Match IDs.xlsx')
    df_matches.rename(columns={'Stadium':'stadium_name','Match ID':'game_id', 'Team A':'home_team_name',
                               'Team B':'away_team_name','Kick-off':'kickoff_time','Match Date':'date',
                               'Group':'tournament_stage','Competition Name':'tournament_name',
                               'Team A 90 score':'home_score_90','Team B 90 score':'away_score_90',
                               'Team A AET score':'home_score_aet','Team B AET score':'away_score_aet',
                               'Team A PK':'home_score_pk','Team B PK':'away_score_pk',
                               'No.':'game_number', 'Competition ID':'tournament_id'
                               },inplace=True)
    df_matches = df_matches.astype({'game_id':'str','kickoff_time':'str'}, errors='raise')


    df_game_info = {}
    for row in df_matches.itertuples():
        if pd.isnull(row.home_score_90):
            continue

        if game_id is not None and row.game_id != game_id:
            continue

        home_team = row.home_team_name
        away_team = row.away_team_name
        home_acronym = map_acronym[home_team]
        away_acronym = map_acronym[away_team]

        home_players = df_players.query("team_name == @home_team").copy()
        home_players['home_away_ref'] = 'home'
        home_players['is_home'] = 1
        away_players = df_players.query("team_name == @away_team").copy()
        away_players['home_away_ref'] = 'away'
        away_players['is_home'] = 0
        game_players = pd.concat([home_players, away_players],ignore_index=True)
        game_players['is_goalkeeper'] = game_players['position'].apply(lambda x: 1 if x.lower() in ('goalie','tw','gk','goalkeeper','keeper') else 0)

        game_players['events_player_id'] = game_players['player_id']
        game_players['events_team_id'] = game_players['team_id']
        game_players['tracking_player_id'] = game_players['player_id']
        game_players['tracking_team_id'] = game_players['team_id']
        game_players['home_team_name'] = home_team
        game_players['away_team_name'] = away_team
        game_players['events_home_id'] = home_players.iloc[0]['team_id']
        game_players['events_away_id'] = away_players.iloc[0]['team_id']
        game_players['events_game_id'] = row.game_id
        game_players['tracking_home_id'] = home_players.iloc[0]['team_id']
        game_players['tracking_away_id'] = away_players.iloc[0]['team_id']
        game_players['tracking_game_id'] = row.game_id
        game_players['home_acronym'] = home_acronym
        game_players['away_acronym'] = away_acronym
        game_players['home_score_90'] = row.home_score_90
        game_players['away_score_90'] = row.away_score_90
        game_players['home_score_aet'] = row.home_score_aet
        game_players['away_score_aet'] = row.away_score_aet
        game_players['extra_time'] = 1 if pd.notnull(row.away_score_aet) else 0
        game_players['home_score_pk'] = row.home_score_pk
        game_players['away_score_pk'] = row.away_score_pk
        game_players['kickoff_time'] = row.kickoff_time
        game_players['date'] = row.date
        game_players['stadium_name'] = row.stadium_name
        game_players['tournament_stage'] = row.tournament_stage
        game_players['tournament_name'] = row.tournament_name
        game_players['tournament_id'] = row.tournament_id
        game_players.drop(columns=['player_id','team_id'], inplace=True)

        df_game_info[row.game_id] = game_players
    
    return df_game_info[game_id] if game_id is not None else df_game_info
