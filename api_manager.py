import time
import ast
import requests
import os
import numpy as np
import json

class apiManager(object):

	def __init__(self, argv=None):
		print(argv)
		self.tst_cid = os.environ['AZURE_CLIENT_ID']
		self.tst_sec = os.environ['AZURE_CLIENT_SECRET']
		self.prd_cid = os.environ['PRD_AZURE_CLIENT_ID']
		self.prd_sec = os.environ['PRD_AZURE_CLIENT_SECRET']
		self.tst_prd = argv.tst_prd

	def get_auth_token(self):
		print('getting auth token')
		headers = {
		    'client_id': self.tst_cid if self.tst_prd == 'tst' else self.prd_cid,
		    'client_secret': self.tst_sec if self.tst_prd == 'tst' else self.prd_sec,
		    # 'Content-Length': '0',
		    'Content-Type': 'application/x-www-form-urlencoded',
		}
		url = 'https://fdh-tst.fifa.org/api/v1/generatetoken/oauth2/v2.0/token' if self.tst_prd == 'tst' else 'https://fdh.fifa.org/api/v1/generatetoken/oauth2/v2.0/token'
		response = requests.post(url, headers=headers)
		j = response.json()
		self.token = j['access_token']

	def get_req(self, url):
		headers = {
		    "Authorization": "Bearer " + self.token
		}
		print(f"Making req (not printing headers): {url}")
		response = requests.get(url, headers=headers)
		if response.status_code == 200:
			if 'chyronhegopassthroughdata' in url:
				print('Getting raw tracking but sleeping first')
				time.sleep(2)
				response = requests.get(url, headers=headers)
				# print(response.content[:100])
				return response.content
			else:
				j = response.json()
				return j
		elif response.status_code == 202:
			print(response.text[:100])
			time.sleep(10)
			return(self.get_req(url))
		else:
			print(response.status_code)
			print(response.text)
			exit()
		

	def get_competitions(self):
		url = "https://api-fdh-prd.fifa.org/api/v1/fdh_pg/ifes_competitions_lookup"
		competitions = self.get_req(url)
		print(competitions)

	def get_schedule(self, competitionId):
		if self.tst_prd == 'tst':
			url = "https://fdh-tst.fifa.org/api/v1/fdh_pg/ifes_competition_schedules?competitionId=eq." + competitionId
		else:
			url = "https://api-fdh-prd.fifa.org/api/v1/fdh_pg/ifes_competition_schedules?competitionId=eq." + competitionId
		schedule = self.get_req(url)
		for g in schedule:
			print(f"{g['competitionStage']} - {g['teamBNameShort']} vs {g['teamANameShort']} on {g['matchDate']} ID: {g['matchId']}")
		return schedule

	def get_t05(self, gameId, dataQuality):
		if self.tst_prd == 'tst':
			url = "https://fdh-tst.fifa.org/api/v1/chyronhegopassthrough/Tf05?GameID={_game_id_}&DataQuality={_data_quality_}".format(_game_id_ = gameId, _data_quality_=dataQuality)
		else:
			url = "https://fdh.fifa.org/api/v1/chyronhegopassthrough/Tf05?GameID={_game_id_}&DataQuality={_data_quality_}".format(_game_id_ = gameId, _data_quality_=dataQuality)
		t05 = self.get_req(url)
		# print(t05)
		return t05

	def get_t08(self, gameId, dataQuality):
		if self.tst_prd == 'tst':
			url = "https://fdh-tst.fifa.org/api/v1/chyronhegopassthrough/Tf08?GameID={_game_id_}&DataQuality={_data_quality_}".format(_game_id_ = gameId, _data_quality_=dataQuality)
		else:
			url = "https://fdh.fifa.org/api/v1/chyronhegopassthrough/Tf08?GameID={_game_id_}&DataQuality={_data_quality_}".format(_game_id_ = gameId, _data_quality_=dataQuality)
		t08 = self.get_req(url)
		# print(f't08:{t08}')
		return t08

	def get_raw_tracking(self, gameId, dataQuality):
		if self.tst_prd == 'tst':
			url = "https://fdh-tst.fifa.org/api/v1/chyronhegopassthroughdata/RawData?GameID={_game_id_}&DataQuality={_data_quality_}".format(_game_id_ = gameId, _data_quality_=dataQuality)
		else:
			url = "https://fdh.fifa.org/api/v1/chyronhegopassthroughdata/RawData?GameID={_game_id_}&DataQuality={_data_quality_}".format(_game_id_ = gameId, _data_quality_=dataQuality)
		raw_data = self.get_req(url)
		return raw_data

	def get_game_metadata(self, gameId, dataQuality):
		if self.tst_prd == 'tst':
			url = "https://fdh-tst.fifa.org/api/v1//chyronhegopassthroughdata/GameMetadata?GameID={_game_id_}&DataQuality={_data_quality_}".format(_game_id_ = gameId, _data_quality_=dataQuality)
		else:
			url = "https://fdh.fifa.org/api/v1//chyronhegopassthroughdata/GameMetadata?GameID={_game_id_}&DataQuality={_data_quality_}".format(_game_id_ = gameId, _data_quality_=dataQuality)
		game_metadata = self.get_req(url)
		print(f"game_metadata: {game_metadata}")
		try:
			return_me = ast.literal_eval(game_metadata.decode('utf-8'))
		except:
			return_me = json.loads(game_metadata.decode('utf-8'))
		return return_me

	def get_player_position(self, game_id):
		url = "https://fdh-tst.fifa.org/api/v1/fdp/tactical_lineups?matchId={_game_id_}".format(_game_id_=game_id)
		l = self.get_req(url)
		# print(l)

	def get_tactical_lineup(self, match_id):
		# url = "https://fdh-tst.fifa.org/api/v1/fdp/tactical_lineups?matchId={matchId}".format(matchId = match_id)
		# url = "https://apiportal-fdh.fifa.org/api/v1/fdp/tactical_lineups?matchId={matchId}".format(matchId = match_id)
		# url = "https://api-fdh-prd.fifa.org/api/v1/fdp/tactical_lineups?matchId={matchId}".format(matchId = match_id)
		url = "https://fdh.fifa.org/api/v1/fdp/tactical_lineups?matchId=eq.{matchId}".format(matchId = int(match_id))
		# url = "https://api-fdh-prd.fifa.org/api/v1/fdp/tactical_lineups?matchId=eq.{matchId}".format(matchId = int(match_id))
		tl = self.get_req(url)
		return tl

class dotdict(dict):
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

if __name__ == '__main__':
	am = apiManager(argv=dotdict({"tst_prd":"prd"}))
	am.get_auth_token()
	# tl = am.get_competitions()
	tl = am.get_schedule("255711")
