import pandas as pd
import numpy as np
import xml.etree.ElementTree as et
import sys
from datetime import datetime, timedelta
from football_logic import possession_model, event_detection
from tracking_routines import *
from set_inputs import get_params, load_game_params
from api_manager import apiManager
import os
import json
from azure.storage.filedatalake import DataLakeServiceClient
from azure.core._match_conditions import MatchConditions
from azure.storage.filedatalake._models import ContentSettings
from azure.identity import ClientSecretCredential
from azure.storage.filedatalake import FileSystemClient, DataLakeDirectoryClient
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from game_time import GameTime

goalkeeper_names = ['goalie','TW','GK','goalkeeper','Goalkeeper','gk']

class externalAutoEventer(object):

	def __init__(self, argv=None):
		cred = ClientSecretCredential(
			tenant_id=os.environ['AZURE_TENANT_ID'], client_id=os.environ['AZURE_CLIENT_ID'], client_secret=os.environ['AZURE_CLIENT_SECRET']
		)
		account_url = "https://fdhtstwesa001.dfs.core.windows.net"
		container = "aed"
		self.fs = FileSystemClient(credential=cred, account_url=account_url, file_system_name=container)
		self.game_id = argv.game_id
		self.provider = argv.provider
		self.data_quality = argv.data_quality
		self.dataset = argv.dataset
		self.input_params = {'set_possession_logic':argv.set_possession_logic,
									'postprocess':argv.postprocess,
									'set_event_logic':argv.set_event_logic,
									'compute_modes':argv.compute_modes,
									'compute_pressures':argv.compute_pressures,
									'compute_pitch_control':argv.compute_pitch_control,
									'filter_ball_position':argv.filter_ball_position,
									'filter_ball_speed':argv.filter_ball_speed,
									'filter_player_speed':argv.filter_player_speed,	
									'max_halves':argv.max_halves								
									}
		self.am = apiManager(argv)

		t1 = datetime.now()
		print("Initalized at: " + str(t1))

	

	def format_player_info(self, player, is_home, team_name, team_id):
		this_player = {
			'isHome': is_home
			, 'sTeamName': team_name
			, 'iTeamID': str(team_id)
			, 'iPlayerId': player['PlayerID']
			, 'sPlayerName': player['PlayerName']
			, 'iJersey': player['Jersey']
		}
		return this_player

	def get_players_df(self, team_info, player_game_info):
		all_players = []

		## player_game_info calls tactical lineups and is used only for positions
		positions_df = player_game_info[['playerId', 'positionName']]
		positions_df = positions_df.rename(columns={"playerId": 'iPlayerId', "positionName": "position"})
		positions_df['iPlayerId'] = positions_df['iPlayerId'].astype(str)
		positions_df['is_goalkeeper'] = positions_df['position'].apply(lambda x: 1 if x in goalkeeper_names else 0)

		## The rest of the player info is extracted from Tracabs T05
		for hp in team_info['HomeTeam']['Players']:
			this_player_info = self.format_player_info(hp, 1, self.home_name, self.home_id)
			all_players.append(this_player_info)
		for ap in team_info['AwayTeam']['Players']:
			this_player_info = self.format_player_info(ap, 0, self.away_name, self.away_id)
			all_players.append(this_player_info)

		players_df = pd.DataFrame(all_players)
		players_df['iPlayerId'] = players_df['iPlayerId'].astype(str)
		players_df['iJersey'] = players_df['iJersey'].astype(str)
		players_df = players_df.merge(positions_df, on = 'iPlayerId', how = 'left')
		players_df = players_df.rename(columns={
				"sPlayerName": "name"
				, "iPlayerId": "tracking_player_id"
				, "iTeamID": "tracking_team_id"
				, "sTeamName":"team_name"
				, "iJersey": "jersey_number"
				, "isHome": "is_home"
			})	
		players_df['home_away'] = players_df['is_home'].apply(lambda x: 'home' if x == 1 else 'away')
		players_df['tracking_home_id'] = self.home_id
		players_df['tracking_away_id'] = self.away_id
		players_df['home_team_name'] = self.home_name
		players_df['away_team_name'] = self.away_name
		print(players_df)
		return players_df


	def get_player_game_info(self):
		player_game_info = self.am.get_tactical_lineup(self.game_id)
		away_player_game_info = pd.DataFrame(player_game_info[0]['payload'][0]['awayTeam']['players'])
		away_player_game_info['isHome'] = 0
		home_player_game_info = pd.DataFrame(player_game_info[0]['payload'][0]['homeTeam']['players'])
		home_player_game_info['isHome'] = 1
		player_game_info = away_player_game_info.append(home_player_game_info)
		return player_game_info



	def main(self):
		t1 = datetime.now()

		params = get_params(self.provider, self.dataset, **self.input_params)

		self.am.get_auth_token()
		game_info = self.am.get_t08(self.game_id, self.data_quality)

		print("game_info")
		print(game_info)
		# This is currently only set up to work with Tracab
		# May not be an issue if only Tracab has tracking data on FDP?
		stadium = game_info['TracabDocument']['ArenaName']

		## Player & team metadata
		self.am.get_auth_token()
		team_info = self.am.get_t05(self.game_id, self.data_quality)

		self.home_id = str(team_info['HomeTeam']['TeamID'])
		self.home_name = team_info['HomeTeam']['TeamName']
		self.away_id = str(team_info['AwayTeam']['TeamID'])
		self.away_name = team_info['AwayTeam']['TeamName']
		
		self.am.get_auth_token()
		player_game_info = self.get_player_game_info()

		# home_goalkeeper_ids, away_goalkeeper_ids = self.get_goalkeeper_ids(player_game_info)

		players_df = self.get_players_df(team_info, player_game_info)

		# players_dict = self.get_players_dict(players_df)

		## Tracking metadata
		self.am.get_auth_token()
		meta_data = self.am.get_game_metadata(self.game_id, 1)
		params['fps'] = 1/float(meta_data['FrameRate'])


		game_params = load_game_params(self.game_id, players_df)
		params.update(game_params)	
		# length = x = long
		params['pitch_length'] = float(meta_data['PitchLongSide']) / 100.0
		# width = y = short
		params['pitch_width'] = float(meta_data['PitchShortSide']) / 100.0


		halves_df = get_halves_tracab(params, meta_data)

		print("Periods df created at: " + str(datetime.now()))


		self.am.get_auth_token()
		tracking_data = self.am.get_raw_tracking(self.game_id, 1)
		print(tracking_data[:100])
		tracking_data = tracking_data.decode('utf-8').split(':\n')

		print(str(len(tracking_data)) + " total frames to sort out...")
		print("Tracking data read at: " + str(datetime.now()))

		i = 0
		print("starting_frame_loop_at:" + str(datetime.now()))
		for this_half in halves_df.itertuples():

			this_half = this_half._asdict()

			# Set parameters
			segment_id = str(self.game_id) +'_'+str(this_half['half'])
			params['segment_id'] = 	segment_id
			params['half']	= int(this_half['half'])				

        	# Parse tracking data
			formatted_tracking_dict, max_i = parse_half_tracab(this_half, tracking_data, params, i)

			i += max_i
						
			this_segment = {}
			print("Period: " + str(this_half['half']) + " trimmed at: " + str(datetime.now()))
			
			this_segment['params'] = prepare_tracking_data(formatted_tracking_dict, params)			

			del formatted_tracking_dict

			####### POSSESSION MODEL #######
			print("Ready to run events...done preprocessing half" + str(this_half['half']))
			t2 = datetime.now()
			print(t2 - t1)

			this_segment = possession_model(this_segment)

			this_segment = event_detection(this_segment)

			print(f"{self.dataset}/{segment_id}.json")
			
			file_client = self.fs.get_file_client(file_path=f"automated_events/{self.dataset}/{str(self.game_id)}/{segment_id}.json")
			# file_client = self.fs.get_file_client(file_path=f"this_is_a_test.json")
			df = this_segment['models']['events']
			j = df.to_json(orient='records')
			file_client.upload_data(j, overwrite=True)

			file_client.close()

			t3 = datetime.now()
			print("Ending at: " + str(t3))
			print(t3 - t2)

class dotdict(dict):
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

# if __name__ == '__main__':
# 	print('v13')
# 	# argv = {
# 	# 	'game_id': "132284", "data_quality": "1", 'dataset': 'u20_womens_world_cup'
# 	# 	, 'provider': 'tracab4', 'window_ball_position': 13, 'window_ball_speed':7
# 	# 	, 'window_player_speed': 7, 'optimize': False, 'postprocess':True
# 	# }
# 	# argv = {
# 	# 	'game_id': "132284", "data_quality": "1", 'dataset': 'u17_womens_world_cup'
# 	# 	, 'provider': 'tracab4', 'window_ball_position': 13, 'window_ball_speed':7
# 	# 	, 'window_player_speed': 7, 'optimize': False, 'postprocess':True
# 	# }
# 	# argv = {
# 	# 	'game_id': "128083", "data_quality": "1", 'dataset': 'mwc_22'
# 	# 	, 'provider': 'tracab5', 'window_ball_position': 0, 'window_ball_speed':7
# 	# 	, 'window_player_speed': 7, 'set_event_logic': True, 'postprocess':True
# 	# 	, 'set_possession_logic': True, 'compute_modes':False, 'compute_pressures':False
# 	# 	, 'compute_pitch_control':False
# 	# }
# 	argv = {
# 		'game_id': "132284", "data_quality": "1", 'dataset': 'wwc_22'
# 		, 'provider': 'tracab4', 'window_ball_position': 13, 'window_ball_speed':7
# 		, 'window_player_speed': 7, 'set_event_logic': True, 'postprocess':True
# 		, 'set_possession_logic': True, 'compute_modes':False, 'compute_pressures':False
# 		, 'compute_pitch_control':False
# 	}
# 	args = dotdict(argv)
# 	eae = externalAutoEventer(args)
# 	eae.main()