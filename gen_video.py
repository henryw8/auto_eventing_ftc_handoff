import sys
import logging
from datetime import datetime

import pandas as pd
import itertools

import json
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import random
import numpy as np
from time import time
from copy import copy
import os
from utils import decompress_pickle

import ast
import argparse

import itertools

   

def get_args():
    """
    This routine gets the arguments from the command line and parses them    

    :return: Dictionary of arguments
    """ 
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", help = "State name of provider", type=str)
    parser.add_argument("--dataset", help = "Name of dataset", type=str)
    parser.add_argument("--game_id", help = "Game ID", type=str)
    parser.add_argument("--filter_ball_position", help = "Parameters for SG ball position filter", type=tuple)
    parser.add_argument("--filter_ball_speed", help = "Parameters for SG ball speed filter", type=tuple)
    parser.add_argument("--filter_player_speed", help = "Parameters for SG player speed filter", type=tuple)
    parser.add_argument("--half", help = "Halves for which to generate the video", type=int)   
    parser.add_argument("--plot_aux", help = "Plot auxiliary lines of the pitch", type=bool)
    parser.add_argument("--load_events", help = "Load events from a given run", type=str)
    parser.add_argument("--override", help = "Override existing videos", type=int)
    parser.add_argument("--interval", nargs='+', help = "Start and end minutes of video", type=int)
    parser.add_argument("--team_colors", help = "Team colors as a dictionary {'home':color, 'away':color, 'goalkeepers':color}", type=str)
    parser.add_argument("--plot_dir", help = "Plot direction as a vectore", type=int)
    parser.add_argument("--flip_xy", help = "Flip xy coordinates for visualization", type=int)
    parser.add_argument("--load_stat", help = "Show team stat", type=str)
    parser.add_argument("--plot_circles", help = "Plot Possession Zone circles", type=int)
    parser.add_argument("--player_display", help = "How to id players in the event annotations (jersey-team or name)", type=str)
    parser.add_argument("--destination", help = "Name of the destination folder", type=str)


    args = parser.parse_args()
    return args

if __name__ == '__main__': 
    """
    This routine generates a video and runs from the command line. Some sample prompts include
    First half of the WC final: python gen_video.py --provider tracab5 --game_id 128083 --half 1 --dataset mwc_22  
    First five minutes of second half of the WC final: python gen_video.py --provider tracab5 --game_id 128083 --half 2 --interval 0 4 --dataset mwc_22  
    First half of the WC final with player directions and auto-events: python gen_video.py --provider tracab5 --game_id 128083 --half 2 --plot_dir 1 --events '2023-05-10' --dataset mwc_22  
    First minute of the WC final with team colors changed: python gen_video.py --provider tracab5 --game_id 128083 --half 2 --team_colors "{'home':'skyblue','away':'navy', 'goalkeepers':'yellow'}" --dataset mwc_22 --interval 0 0 

    :return: Animation object to be saved to specified path
    """ 

    source_path =  '/Users/fvidal/Documents/TTY/fifa/'
    sys.path.insert(0, source_path+'auto_eventing/')

    from graphics import WholePitchVideo
    from game_time import GameTime, _display_football_time
    from set_inputs import parse_lineups, load_game_params, get_params
    from tracking_routines import *
    from utils import decompress_pickle, compressed_pickle

    print('Imports successful')

    args = get_args()

    dataset = args.dataset
    provider = args.provider    

    metadata_path = source_path + f'Datasets/{dataset}/IFES/'    
    output_path = source_path + f'Datasets/{dataset}/autoevents/'    
    tracking_raw_path = source_path + f'Datasets/{dataset}/tracking/{provider}/'

    if args.plot_aux is not None:
        plot_aux = args.plot_aux
    else:
        plot_aux = False
        
    if args.filter_ball_position is not None:
        filter_ball_position = args.filter_ball_position
    else:
        filter_ball_position = (2,13) if 'tracab4' in provider else (0,0)

    if args.filter_player_speed is not None:
        filter_player_speed = args.filter_player_speed
    else:
        filter_player_speed = (2,7)

    if args.filter_ball_speed is not None:
        filter_ball_speed = args.filter_ball_speed
    else:
        filter_ball_speed = (2,7)    

    if args.flip_xy is None:
        flip_xy = False
    else:
        flip_xy = bool(args.flip_xy)
    

    if args.destination is None:
        destination = 'videos_tracking'
    else:
        destination = args.destination

    if args.interval is not None:
        interval = args.interval
    else:
        interval = None
        
    if args.plot_dir is not None:
        plot_dir = bool(args.plot_dir)
    else:
        plot_dir = False

    if args.plot_circles is not None:
        plot_circles = bool(args.plot_circles)
    else:
        plot_circles = True

    if args.player_display is None:
        player_display = 'jersey_team'
    else:
        player_display = 'name'
        
    if args.half is not None:
        half_to_create = args.half
    else:
        half_to_create = 0            

    if args.team_colors is not None:
        team_colors = ast.literal_eval(args.team_colors)
    else:
        team_colors = {'home':'royalblue','away':'orangered', 'goalkeepers':'mediumseagreen'}

    if provider == 'skillcorner':
        fps = 10
    else:
        fps = 25
        
    list_of_games = [str(s) for s in pd.read_excel(metadata_path+'Match IDs.xlsx')['Match ID'].tolist()]

    
    if args.game_id is None:
        game_id = ''     
        print(f"Creating videos for {len(list_of_games)} matches!!")
    else:
        game_id = args.game_id        
        list_of_games = [game_id]    
    
    # events = dict()  
    suffix = ''  
    if args.load_events is not None:
        suffix += '_events'
    if args.load_stat is not None:
        suffix += f'_{args.load_stat}'
  

    params_base = get_params(provider, dataset, filter_ball_position=filter_ball_position,
                                                filter_ball_speed=filter_ball_speed,
                                                filter_player_speed=filter_player_speed)

    video_args = {'fps':fps, 'team_colors':team_colors, 
                      'plot_aux':plot_aux,'dataset':dataset,
                     'n': 3 if fps >=25 else 1, 
                      'highlight_visible':True if provider=='skillcorner' else False,
                     'plot_dir':plot_dir, 'flip_xy':flip_xy,
                      'plot_circles':plot_circles,
                      'time_from_zero':False,
                      'player_display':player_display}

        
    for i, game_id in enumerate(list_of_games,1):
        
        ####### METADATA ####### 
        params = copy(params_base)
        t0 = time()
        players_df = parse_lineups(metadata_path, game_id)
        game_params = load_game_params(game_id, players_df)
        params.update(game_params)    

        print(f"{params['home_team_name']} vs. {params['away_team_name']} -- ({i}/{len(list_of_games)})")    
            
        ####### TRACKING DATA #######
        t1 = time()
        
        ## Parse tracking data downloaded from FDP  
        gameDate = str(players_df.iloc[0]['date']).replace('-','').split(' ')[0]
        folder_name = f"TRACKING_DATA_{gameDate}_{players_df.iloc[0]['home_acronym']}-{players_df.iloc[0]['away_acronym']}"
        path_to_data = tracking_raw_path+f'{folder_name}/'
        if os.path.exists(path_to_data+f'{game_id}.dat'):                          
            with open(path_to_data+f'{game_id}_metadata.json','r') as g:
                meta_data = json.loads(g.readlines()[0])  
            params['pitch_length'] = float(meta_data['PitchLongSide'])/100
            params['pitch_width'] = float(meta_data['PitchShortSide'])/100

            tracking_data = np.genfromtxt(path_to_data+f'{game_id}.dat',names=True,dtype=None,encoding='UTF-8',delimiter='\n')   
            halves_df = get_halves_tracab(params, meta_data)
        else:
            ### No tracking data found, moving on to the next segment
            print(70*'-')
            continue
        
        ## Loop over halves
        for this_half in halves_df.itertuples():
            
            this_half = this_half._asdict()     
            if half_to_create != 0 and this_half['half'] != half_to_create:
                continue
            params_this_half = copy(params)
            
            ## Initialize output file
            this_segment = {}
            segment_id = f"{game_id}_{this_half['half']}"
            params_this_half['segment_id'] =    segment_id
            params_this_half['half']    = int(this_half['half'])                            
                       
            ## If requested, load autoevents                  
            found = False
            if 'events' in suffix:                
                try:
                    this_segment = decompress_pickle(output_path+f'{args.load_events}/{segment_id}')
                    found = True
                except:         
                    print(f'Events not found for {segment_id}')
                    suffix = ''

            ## Parse tracking data
            print(f"Parsing tracking data for half {this_half['half']}")
            formatted_tracking_dict, _ = parse_half_tracab(this_half, tracking_data, params_this_half)
            params = prepare_tracking_data(formatted_tracking_dict, params_this_half)
            del formatted_tracking_dict
            td = params['td']
            params.pop('td')
            print(70*'-')

            ## Create video only for certain interval
            suffix_game = suffix
            if interval is not None:
                if len(interval) == 1:
                    interval = 2*interval
                td = td[pd.eval("td.minute >= interval[0] and td.minute <= interval[1]")].reset_index(drop=True)
                suffix_game += f"_{interval[0]}-{interval[1]}"

            nametag = f"{segment_id}{suffix_game}.mp4"

            ## Load team stats
            if found:
                events_dict = {'events':this_segment['models']['events'],
                          'params':this_segment['params']}

                if args.load_stat is not None:
                    events_dict['team_stat'] = this_segment['models'][args.load_stat]
                else:
                    events_dict['team_stat'] = pd.DataFrame()
            else:
                events_dict = {}

            ## Launch video creator
            video_args['path'] =  source_path+f'Datasets/{dataset}/{destination}/{nametag}'
            video_args['nametag'] = nametag

            print(f"{game_id} Half {this_half['half']}")                
            anim = WholePitchVideo(td.query(f"half == {this_half['half']}").copy(), players_df, events=events_dict.get('events',pd.DataFrame()),
                                   team_colors=video_args['team_colors'],
                                   plot_aux=video_args['plot_aux'], params=events_dict.get('params',{}),
                                   annotate_tag='jersey_number', 
                                   fps=video_args['fps'], n=video_args['n'],
                                   highlight_visible=video_args['highlight_visible'],
                                   plot_direction=video_args['plot_dir'],
                                   flip_xy=video_args['flip_xy'],
                                   team_stat=events_dict.get('team_stat',pd.DataFrame()),
                                   plot_circles=video_args['plot_circles'],
                                   player_display=video_args['player_display'],
                                   time_from_zero=video_args.get('time_from_zero',True),
                                  verbose=25).get_anim()
            
            t0 = time()
            anim.save(f"{video_args['path']}")
            print(f"Video {game_id} Half {this_half['half']} made in {int((time()-t0)/60)} minutes")
        print(70*'-')
        print(70*'-')




