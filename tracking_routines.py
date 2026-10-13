import numpy as np
import pandas as pd
from time import time
import logging
from game_time import GameTime
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from datetime import datetime

def filter_position(data, polyorder, windowlength, is_ball=False, dt=0.04):
    """
    This routine applies a Savitzky-Golay filter to a position data stream, 
    derive it using a cubic spline and filter it again to compute velocities

    :param data: Dataframe of positions for either ball/player
    :param polyorder: Polynomial order to use in filtering, typically 1-2
    :param windowlength: Odd number that indicates the number of neighboring points used for filtering
    :param is_ball: Boolean to identify if ball is filtered
    :return: Updated dataframe with filtered velocities
    """ 

    if windowlength == 0:
        return data

    if is_ball:
        in_game = data.query("ball_status == 'in_play'").reset_index()
        data['ball_speed'] = 0
        for k, segment in in_game.groupby(in_game['index'] - np.arange(in_game.shape[0])):

            if segment.shape[0] == 1:
                continue
            index_fill = segment['index'].tolist()    
            t=segment.time_ms/1000

            if segment.shape[0] >= windowlength and windowlength > 0:

                # Smooth trajectories
                x_sg=savgol_filter(segment.ball_x, windowlength, polyorder)
                y_sg=savgol_filter(segment.ball_y, windowlength, polyorder)

                data.loc[index_fill,'ball_x'] = x_sg
                data.loc[index_fill,'ball_y'] = y_sg

    else:
        
        x = data.x.to_numpy()
        y = data.y.to_numpy()
        t = data.time_ms.to_numpy()/1000
        
        if len(x) >= windowlength and windowlength > 0:
            x_sg=savgol_filter(x, windowlength, polyorder)
            y_sg=savgol_filter(y, windowlength, polyorder)
            data['x'] = x_sg
            data['y'] = y_sg
            
    return data

def compute_velocities(data, polyorder, windowlength, is_ball=False, dt=0.04):
    """
    This routine applies a Savitzky-Golay filter to a position data stream, 
    derive it using a cubic spline and filter it again to compute velocities

    :param data: Dataframe of positions for either ball/player
    :param polyorder: Polynomial order to use in filtering, typically 1-2
    :param windowlength: Odd number that indicates the number of neighboring points used for filtering
    :param is_ball: Boolean to identify if ball is filtered
    :return: Updated dataframe with filtered velocities
    """ 

    if is_ball:
        in_game = data.query("ball_status == 'in_play'").reset_index()
        data['ball_speed'] = 0
        for k, segment in in_game.groupby(in_game['index'] - np.arange(in_game.shape[0])):

            if segment.shape[0] == 1:
                continue
            index_fill = segment['index'].tolist()    
            t=segment.time_ms/1000

            try:
                if segment.shape[0] < windowlength or windowlength == 0:

                    dx = CubicSpline(t, segment.ball_x).derivative()(t)
                    dy = CubicSpline(t, segment.ball_y).derivative()(t)

                else:
                    # Smooth trajectories
                    x_sg=savgol_filter(segment.ball_x, windowlength, polyorder)
                    y_sg=savgol_filter(segment.ball_y, windowlength, polyorder)

                    # Obtain velocities by spline and smooth velocities
                    dx_raw = CubicSpline(t, x_sg).derivative()(t)
                    dx = savgol_filter(dx_raw, windowlength,polyorder)

                    dy_raw = CubicSpline(t, y_sg).derivative()(t)
                    dy = savgol_filter(dy_raw, windowlength,polyorder)

            except:
                dx = segment.ball_x.diff()/dt
                dy = segment.ball_y.diff()/dt

            data.loc[index_fill,'ball_speed'] = np.sqrt(dx**2+dy**2)
    else:
        
        x = data.x.to_numpy()
        y = data.y.to_numpy()
        t = data.time_ms.to_numpy()/1000
        
        try:
            if len(x) < windowlength or windowlength == 0:
                dx = CubicSpline(t, x).derivative()(t)
                dy = CubicSpline(t, y).derivative()(t)                

            else:
                x_sg=savgol_filter(x, windowlength, polyorder)
                y_sg=savgol_filter(y, windowlength, polyorder)
                dx_raw = CubicSpline(t, x_sg).derivative()(t)
                dy_raw = CubicSpline(t, y_sg).derivative()(t)

                maxspeed = 12
                raw_speed = np.sqrt( dx_raw**2 + dy_raw**2 )
                dx_raw[ raw_speed>maxspeed ] = np.nan
                dy_raw[ raw_speed>maxspeed ] = np.nan

                dx = savgol_filter(dx_raw, windowlength,polyorder)
                dy = savgol_filter(dy_raw, windowlength,polyorder)

        except:
            dx = data.x.diff()/dt
            dy = data.y.diff()/dt

        data['speed'] = np.sqrt(dx**2+dy**2)
        data['vx'] = dx
        data['vy'] = dy

    return data

def prepare_tracking_data(td_all, params):

    """
    This routine prepares the format tracking data to be used for auto-eventing, including filtering positions and computation of velocities

    :param td_all: Dictionary containing the tracking data for the players and the ball  
    :param params: Dictionary containing all the relevant parameters    
    :return: Params dictionary with updated tracking data
    """ 

    players_df = params['players_df']
    home_id = params['home_id']
    away_id = params['away_id']
    td_players = td_all['players']
    td_ball = td_all['ball']     

    td_players['team_id'] = td_players['player_id'].map(players_df['tracking_team_id'].to_dict())
    td_players['jersey_number'] = td_players['player_id'].map(players_df['jersey_number'].to_dict())
    td_players['is_home'] = td_players['player_id'].map(players_df['is_home'].to_dict())
    td_ball['half'] = params['half']
    td_ball['minute'] = (td_ball['time_ms']//60000).astype('int')
    td_ball['sec'] = td_ball['time_ms']% (60 * 1000) / 1000

    td = td_players.merge(td_ball, on='frame')
    del td_players, td_ball

    td = trim_tracking_data(td)
    params['active_goal'] = get_active_goal(td, home_id, away_id, params['pitch_length'], params['pitch_width'])

    td['active_goal'] = td['team_id'].map({home_id:np.sign(params['active_goal'][home_id][0]),
                                        away_id:np.sign(params['active_goal'][away_id][0])})  

    columns = ['frame','half','ball_x','ball_y','ball_z','time_ms','ball_status']

    ball_data = td[columns].drop_duplicates(subset='frame').sort_values(by='frame',
                                                                      ascending=True).reset_index(drop=True).copy()

    
    ball_data['game_time'] = ball_data.apply(lambda x: GameTime.from_event_time(x['half'], x['time_ms']), axis=1)
    time_data = ball_data['game_time'].tolist()
    half = td.at[0,'half']
    xy_ball = ball_data[['ball_x','ball_y']].to_numpy()
    ball_status = ball_data['ball_status'].tolist()
    # ball_status = ['dead' if status == 'dead' or np.isnan(xy_ball[i,0]) else 'in_play' for i, status in enumerate(ball_status)]
    ball_data['ball_status'] = ball_status
    
    ## First compute the velocity of the ball regardless of whether the positions needs to be filtered or not
    filter_ball_speed = params.get('filter_ball_speed',(0,0))
    ball_data = compute_velocities(ball_data, params['filter_ball_speed'][0], 
                                    params['filter_ball_speed'][1], is_ball=True, dt=1/params['fps'])
    
    ## Filter position of ball if requested
    filter_ball_pos = params.get('filter_ball_position',(0,0))
    ball_data = filter_position(ball_data, filter_ball_pos[0], 
                                     filter_ball_pos[1], is_ball=True)

    ball_data['ball_displacement'] = np.sqrt(ball_data.ball_x.diff().to_numpy()**2 + ball_data.ball_y.diff().to_numpy()**2)
    td['distance'] = np.sqrt((td.x.to_numpy() - td.ball_x.to_numpy())**2 + (td.y.to_numpy() - td.ball_y.to_numpy())**2)

    td.drop(columns=['ball_x','ball_y','ball_status'], inplace=True)
    td = td.merge(ball_data[['ball_x','ball_y','frame','ball_displacement','ball_status']], on='frame')

    ball_distances = td[['frame',
                       'player_id',
                       'distance']].query("distance < 5").groupby("frame").apply(lambda p: dict(zip(p.player_id, p.distance))).reset_index().rename(columns={'index':'frame',
                       0:'distances'})

    ball_data = ball_data.merge(ball_distances, on='frame', how='left')
    

    ## Compute player velocities
    td_new = []
    filter_player_speed = params.get('filter_player_speed',(0,0))
    for player in td['player_id'].unique():
        player_td = td[pd.eval("td.player_id == player")].copy()
        td.drop(index=player_td.index.tolist(), inplace=True)
        new_player = compute_velocities(player_td, filter_player_speed[0], 
                                        filter_player_speed[1], is_ball=False, dt=1/params['fps'])

        td_new.append(new_player)    
    td = pd.concat(td_new, ignore_index=True).sort_values(by=['frame','team_id'], ascending=True) 
    del td_new  
    

    params['td'] = td
    params['td_frame'] = dict(tuple(td[['frame','player_id','vx','vy','x','y','speed']].groupby('frame')))
    params['ball_data'] = ball_data

    return params


def trim_tracking_data(td):
    """
    This routine cut end of tracking data if some/all players are missing

    :param td: Tracking data
    :return: Updated tracking data
    """     
    max_minute = td.minute.max()-1
    home_ct = td[pd.eval("td.is_home == 1 & td.minute >= max_minute")][['x','frame']].groupby('frame', sort=False).agg('count').reset_index()
    last_frame_home = home_ct.loc[home_ct['x']>= 7,'frame'].iloc[-1]
    away_ct = td[pd.eval("td.is_home == 0 & td.minute >= max_minute")][['x','frame']].groupby('frame', sort=False).agg('count').reset_index()
    last_frame_away = away_ct.loc[away_ct['x']>= 7,'frame'].iloc[-1]
    last_frame = min(last_frame_home, last_frame_away)
    td = td[pd.eval("td.frame <= last_frame")]
    
    return td.reset_index(drop=True)


def get_halves_tracab(params, meta_data):
    """
    This routine returns the start frame and end frame for each half, up to max_halves to handle tournaments without extra time

    :param params: Dictionary containing all the relevant parameters    
    :param metadata: Tracab metadata file parsed as a dict
    :return: Dataframe with one row per half
    """ 
    halves = []
    game_start_frame = None
    for half in range(1, params['max_halves']+1):
        if int(meta_data['Phase' + str(half) + "StartFrame"]) != 0:
            if game_start_frame == None or int(meta_data['Phase' + str(half) + "StartFrame"]) < game_start_frame:
                game_start_frame = int(meta_data['Phase' + str(half) + "StartFrame"])
            this_half = {
                'half': int(half)
                , 'first_frame': int(meta_data['Phase' + str(half) + "StartFrame"])
                , 'last_frame': int(meta_data['Phase' + str(half) + "EndFrame"])
            }
            halves.append(this_half)
    halves_df = pd.DataFrame(halves)
    return halves_df


def get_active_goal(td, home_id, away_id, pitch_length, pitch_width):
    """
    This routine computes the active goals for each team, namely the goal they are trying to score on

    :param td: Dataframe of the tracking data
    :param home_id: String of the home team ID   
    :param away_id: String of the away team ID   
    :param pitch_length: Long dimension of the pitch   
    :param pitch_width: Short dimension of the pitch   
    :return: Dictionary with the active goal for each team ID
    """ 

    home_initial = td.query("time_ms < 1000 and team_id==@home_id")['x'].mean(skipna=True)
    away_initial = td.query("time_ms < 1000 and team_id==@away_id")['x'].mean(skipna=True)

    active_goal = dict()
    if home_initial < away_initial:
        active_goal[home_id] = (pitch_length/2,0)
        active_goal[away_id] = (-pitch_length/2,0)
    else:
        active_goal[home_id] = (-pitch_length/2,0)
        active_goal[away_id] = (pitch_length/2,0)

    return active_goal


def parse_half_tracab(half_info, data, params, i=0):

    """
    This routine parses the raw Tracab data for one half of one game. 
    :param half_info: Information on the half
    :param data: Raw tracking data in supplied by Tracab
    :param params: Dictionary containing all the relevant parameters        
    :param i: Integer to monitor progression of tracking data parsing   
    :return: Dictionary containing a player tracking dataframe (frame, player_id, x, y , speed, time_ms) and a ball tracking
           dataframe (frame, ball_x, ball_y, ball_z, ball_speed, ball_status), value of progression index i 
    """ 

    ball_tracking_array = []
    player_tracking_array = []
    players_dict = params['players_dict']
    for index, f in enumerate(data):
        if type(f) is not str:
            f = f[0]
        if f in ["", None]:
            continue
        i += 1
        info = f.split(':')
        frame = int(info[0])
        if i % 10000 == 0 and params['verbose']:
            print(str(i) + " at " + str(datetime.now()))

        if frame < int(half_info['first_frame']) or frame > int(half_info['last_frame']):
            continue

        time_s = round(int(frame-half_info['first_frame'])/params['fps'],2)
        time_ms = round(1000*time_s)

        players = info[1].split(';')
        for player in players:
            if player == '':
                continue
            player_info = player.split(',')
            jersey_number = str(player_info[2])
            if int(player_info[0]) not in [0, 1]:
                continue
            home_away = "away" if int(player_info[0]) == 0 else "home"
            player_id = str(players_dict[home_away][jersey_number]['tracking_player_id'])
            player_x = float(player_info[3])/100
            player_y = float(player_info[4])/100
            player_speed = float(player_info[5])/3.6
            player_tracking_array.append([frame, player_id, player_x, player_y, player_speed]) 
            del player_id

        ball_info = info[2].split(',')
        ball_x = float(ball_info[0])/100
        ball_y = float(ball_info[1])/100
        ball_z = float(ball_info[2])/100
        ball_speed = float(ball_info[3])/3.6
        ball_status = ball_info[5].replace(';','')

        ball_tracking_array.append([frame, time_ms, ball_x, ball_y, ball_z, ball_speed, ball_status])

    df_players = pd.DataFrame(player_tracking_array, columns=['frame','player_id','x','y','speed'])  
    del player_tracking_array

    df_ball = pd.DataFrame(ball_tracking_array, columns=['frame','time_ms','ball_x','ball_y','ball_z','ball_speed','ball_status'])  
    del ball_tracking_array

    df_ball['ball_status'] = df_ball['ball_status'].replace({'Alive':'in_play',
                                                     'Dead':'dead'})


    df_players = df_players.astype({'frame':'int64','x':'float64','y':'float64','speed':'float64'}, errors='raise')
    df_ball = df_ball.astype({'frame':'int64','time_ms':'int64','ball_x':'float64','ball_y':'float64',
                                    'ball_z':'float64','ball_speed':'float64','ball_status':'str'}, errors='raise')

        
    return {'players':df_players, 'ball':df_ball}, i 


    

