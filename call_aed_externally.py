from external_test import externalAutoEventer
from api_manager import apiManager

# {'competitionId': 283693, 'competitionName': "FIFA U-17 Women's World Cup India 2022™", 'beginDate': '2022-10-11', 'endDate': '2022-10-30'}
# {'competitionId': 283933, 'competitionName': 'FIFA U-20 Women’s World Cup Costa Rica 2022™', 'beginDate': '2022-08-10', 'endDate': '2022-08-28'}

competition_id = "255711"

configs = {
	"283933": {
			"data_quality": "1", 'dataset': 'u20wwc_22'
			, 'provider': 'tracab4', 'filter_ball_position': (2,13), 'filter_ball_speed':(2,7)
			, 'filter_player_speed': (2,7), 'tst_prd': 'prd', 'set_event_logic': True, 'postprocess':True
			, 'set_possession_logic': True, 'compute_modes':False, 'compute_pressures':False
			, 'compute_pitch_control':False
			, 'max_halves': 4
		}
	, "283693": {
			"data_quality": "1", 'dataset': 'u17wwc_22'
			, 'provider': 'tracab4', 'filter_ball_position': (2,13), 'filter_ball_speed':(2,7)
			, 'filter_player_speed': (2,7), 'tst_prd': 'prd', 'set_event_logic': True, 'postprocess':True
			, 'set_possession_logic': True, 'compute_modes':False, 'compute_pressures':False
			, 'compute_pitch_control':False
			, 'max_halves': 2
		}
	, "255711": {
			"data_quality": "1", 'dataset': 'mwc_22'
			, 'provider': 'tracab5', 'filter_ball_position': (0,0), 'filter_ball_speed':(2,7)
			, 'filter_player_speed': (2,7), 'tst_prd': 'prd'
			, 'max_halves': 4, 'set_event_logic': True, 'postprocess':True
			, 'set_possession_logic': True, 'compute_modes':True, 'compute_pressures':True
			, 'compute_pitch_control':True
		}
	, "285026": {
			"data_quality": "1", 'dataset': 'wwc_23'
			, 'provider': 'tracab4', 'filter_ball_position': (2,7), 'filter_ball_speed':(2,7)
			, 'filter_player_speed': (2,7), 'tst_prd': 'prd'
			, 'max_halves': 4, 'set_event_logic': True, 'postprocess':True
			, 'set_possession_logic': True, 'compute_modes':False, 'compute_pressures':False
			, 'compute_pitch_control':False
		}
}

class dotdict(dict):
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

# 131909

matches = ['131909', '134411', '134412']
matches = ['128073']

am = apiManager(argv=dotdict({"tst_prd":"prd"}))
am.get_auth_token()
schedule = am.get_schedule(competition_id)
# print(schedule)
# exit()
# for m in schedule[:5]:
i = 0
for m in schedule:

	# print(f"{m['matchId']}")
	# if i == 0:
	# 	print('running')
	# 	i += 1
	# print(m)
	# print(m['matchId'])

	if str(m['matchId']) in matches:

		print(m['matchId'])

		this_match_config = configs[competition_id]
		this_match_config['game_id'] = m['matchId']
		args = dotdict(this_match_config)
		print(args)
		eae = externalAutoEventer(args)
		eae.main()
	else:
		print('skipping')



# u17 WWC - India 283693 MAX PERIODS: 2

# {'dataset': 'u17wwc_22', 'provider': 'tracab4', 'filter_ball_position': 13, 'filter_ball_speed':7, 'filter_player_speed': 7, 'optimize': False, 'postprocess':True}


# u20 WWC - Costa Rica:283933

# {'dataset': 'u20wwc_22', 'provider': 'tracab4', 'filter_ball_position': 13, 'filter_ball_speed':7, 'filter_player_speed': 7, 'optimize': False, 'postprocess':True}

# mwc Qatar - 255711

# {'competitionId': 283933, 'generalEventId': 283932, 'competitionName': 'FIFA U-20 Women’s World Cup Costa Rica 2022™', 'competitionIfesName': "W-WC U-20 > 2022 > FIFA U-20 Women's World Cup", 'competitionStage': 'Final Stage', 'competitionGroup': 'Play-off for third place', 'matchId': 132295, 'matchNumber': 31, 'matchDate': '2022-08-28', 'kickOffTimeLocal': 1630, 'kickOffTimeUtc': '2022-08-28T22:30:00+00:00', 'teamAId': 1888584, 'teamANameLong': 'Netherlands U20 (women)', 'teamANameShort': 'Netherlands', 'teamAPlaceholderId': 29950, 'teamAPlaceholderName': 'RU29', 'teamACountryCode': 'NED', 'teamBId': 1888622, 'teamBNameLong': 'Brazil U20 (women)', 'teamBNameShort': 'Brazil', 'teamBPlaceholderId': 29951, 'teamBPlaceholderName': 'RU30', 'teamBCountryCode': 'BRA', 'stadiumId': 34705, 'stadiumName': 'Estadio Nacional de Costa Rica', 'city': 'SAN JOSE', 'countryCode': 'CRC', 'timeZone': 'America/Costa_Rica'}