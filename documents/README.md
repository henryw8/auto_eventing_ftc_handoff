# Auto-eventing code

The auto-eventing code is hosted in this [Github repo](https://github.com/rsbensaid/auto_eventing/tree/main). The diagram of the code can be found in the file autoeventing-flowchart.pdf. The explanation of the algorithm may be found in extensive detail in the [auto-eventing paper](https://link.springer.com/article/10.1007/s12283-022-00381-6), where the logic behind the two main functions (`possession_model` and `event_detection`) is thoroughly described.

There are two separate ways of executing the code to compute events:
- Using locally downloaded tracking data (via Jupyter Notebook)
- Fetching data from the FIFA Data Hub (via API)

## Locally downloaded tracking data
The main script is the Jupyter notebook `local_autoeventer.ipynb`, where the auto-eventing code is executed locally and the output is stored in a compressed pickle format. 

The tracking data can be downloaded manually from the FDP, and the first step in the auto-eventing process is to ingest the tracking data from its raw format to a tabular format, using the `tracking_routines.py` module. We have only included code to parse Tracab data because that is the only one available in the FDP, but it can work analogously for other data providers.

In addition to the tracking data, the player metadata is needed to execute the pipeline, namely the mapping from player ID to team ID, jersey number, player name and the set of goalkeeper IDs. The metadata has been made available to us in the past by Régis Bardet via direct transfer of the IFES Match IDs, Player IDs, Team IDs and Referee IDs Excel files. 

In order to execute the code, you need to change the Git path to your local path, as well as the path where the tracking data/IFES metadata is located and where the output will be written.

## API calls to FIFA Data Hub
This section will be populated shortly.

## Hyperparameters
These are the hyperparameters that control how passes and possessions are determined in the auto-eventing pipeline, and hence have the most impact in the results. They are defined separately from the rest of parameters to perform hyperparameter optimization more effectively in the presence of tracking data from providers for which we do not know the optimal settings.

| Hyperparameter | Definition |
| ------ | ------ |
| `ball_change_direction` | Change in ball direction, in degrees |
| `ball_change_speed` |  Change in ball speed, in m/s |
| `min_ball_displacement` | Change in minimum ball displacement, in m |
| `continued_possession` | Boolean, if 0 then the player to define continued possessions, see th `set_gains_losses` on `football_logic.py` |
| `pz` | Radius of possession zone, in m |
| `pz_gk` | Radius of possession zone for goalkeepers, in m |
| `pz_contest` | Radius of contest zone for the ball, in m |

## Parameters
These are the remaining parameters and tolerances that control several aspects of the auto-eventing pipeline. The parameters are grouped in sections according to their nature.

### General parameters
Provider information and auto-eventing algorithm options
| Parameter | Definition |
| ------ | ------ |
| `provider` |  Name of tracking data provider. |
| `dataset` |  Name of dataset. |
| `fps` | Frame rate of tracking data. |
| `verbose` | Boolean to display auto-eventing info & warnings on screen. |
| `set_event_logic` | Boolean to execute the function that adds temporal logic to events. | 
| `set_possession_logic` | Boolean to execute the function that chains events to the previous and next event, as well as adds the outcome of events and incorporates crosses. | 
| `postprocess` | Boolean to execute the function that adds contextual attributes to events, such as angle, team x-y span, player location with respect to other team, players overcome by passes, distances traveled by players & ball from pass to reception. |
| `compute_modes` | Boolean to execute the function that computes the defensive modes. |
| `compute_pressures` | Boolean to execute the function that computes the individual and team pressures. |
| `compute_pitch_control` | Boolean to execute the function that computes the pitch control gain and passing lines for every uncontested pass. |
| `pitch_measures` | Dictionary of pitch measures, including penalty mark, goal area, goal, penalty area and central arc, in m.|
| `active_goal` | Dictionary where the keys are the home/away ID and the values are the x-y coordinates of the goal they are attacking on the current segment.|
           
### Game data
Game-specific data and metadata
| Parameter | Definition |
| ------ | ------ |
| `game_id` |  ID of the game, string. |
| `half` |  Half of the game to evaluate, takes values 1-2 if regular time and 3-4 if extra time, integer.  |
| `segment_id` | Concatenation of `game_id` and `half`, string. |
| `td` | Tracking data as the output of `preapre_tracking_data`. The center mark corresponds to (0,0).  |
| `td_frame` | Tracking data rearranged as a dictionary and indexed by frames for faster lookup. |
| `ball_data` | Ball tracking data, including displacement, changes in speed and direction.  |
| `home_id` | ID of home team, string. |
| `away_id` | ID of away team, string. |
| `home_team_name` | Name of home team, string. | 
| `away_team_name` | Name of away team, string. | 
| `home_goalkeeper_ids` | Player IDs corresponding to goalkeepers in the home team, list of strings.|
| `away_goalkeeper_ids` | Player IDs corresponding to goalkeepers in the away team, list of strings.|
| `goalkeeper_ids` | Player IDs corresponding to all goalkeepers, list of strings.|
| `players_df` | Dataframe containing one row per player in each team, indexed by the player ID. The columns should contain at least team ID, team name, home/away boolean, player name, jersey number, player position.|
| `players_dict` | Dictionary with `home`/`away` keys and values a dictionary of all players in a team indexed by `jersey_number`.|
| `pitch_length` | Long dimension of pitch, float in m. |
| `pitch_width` | Short dimension of pitch, float in m. |
| `max_periods` | Maximum number of halves, 2 if match cannot have extra time and 4 if match can have extra time.|
    
    
### Tolerances
Tolerances to account for imperfect tracking data or heuristics.

| Parameter | Definition |
| ------ | ------ |
| `pz_setpiece` | Radius of possession zone for taking a set piece. Typically `pz_setpiece` > `pz` because the in-play/dead boolean may be inexact and thus accounts for some buffer in case the ball has already been kicked when the boolean turns to in-play. |
| `player_in_pitch` |  Dictionary of tolerances to consider whether a player is inside the pitch, in m. |
| `ball_in_pitch` |  Tolerance to consider whether the ball is inside the pitch, in m. |
| `in_area` | Tolerance to consider whether a player is player inside penalty area. |
`kickoff_player_out_pitch` | Tolerance to exclude players that are near the borders of the pitch at kick-off, in m. |
| `on_target_threshold` | Tolerance for shot on target, in m. |
| `save_deflect_interval` | Duration to consider that a player possession after a goalkeeper save comes from an unintentional deflect and not an intentional pass, in s. |
| `cross_distance` | Minimum distance to consider a pass to be a cross, in m. |
| `cross_height` | Minimum height at any point in trajectory to consider a pass to be a cross, in m. |
| `goal_var` | Heuristic to control the proximity of the ball to the penalty box towards detecting goals awarded late by VAR as per `set_score`, in m.|
| `tol_ck` | Tolerance for corner kick, in m, visually explained in the paper.|
| `tol_gk` | Tolerance for goal kick, in m, visually explained in the paper.|
| `tol_ti` | Tolerance for throw-in, in m, visually explained in the paper.|
| `tol_pk1` | Tolerance 1 for penalty kick, in m, visually explained in the paper.|
| `tol_pk2` | Tolerance 2 for penalty kick, in m, visually explained in the paper.|
| `tol_pk3` | Tolerance 3 for penalty kick, in m, visually explained in the paper.|
| `tol_ko1` | Tolerance 1 for kick-off, in m, visually explained in the paper.|
| `tol_ko2` | Tolerance 2 for kick-off, in m, visually explained in the paper.|
| `kickoff_ball_max_displacement` | Tolerance to control maximum ball displacement allowed in a given interval to identify a kickoff, in m. Useful in case in-play/dead boolean is not accurate to prevent missing kickoffs. |
| `set_piece_max_displacement` | Tolerance to control the maximum displacement both the ball and the player are allowed between the play resuming from a set piece and the first pass being made, in m.|
| `dead_interval_ball_max_displacement` | Tolerance to control the maximum displacement the ball is allowed to have between consecutive dead ball intervals, in m. If displacement is less than tolerance, then we consider the two dead ball intervals are the same and that ball status should have stayed dead instead of making the transition dead->in-play->dead, therefore we collapse both intervals into a single one. |
| `aerial_z` | Minimum height of ball to consider a duel is aerial, in m.|
|`gk_z_claim`| Height above which a GK reception is considered a claim, in m. |
| `event_logic_lookbehind` | Maximum lookbehind time to change the last touch before out-of-bounds, in s, used in `set_event_logic`. |
| `db_from_sp` | Dictionary of dead ball event - set piece event correspondence. |

### Tracking data filtering
Parameters to handle filtering of positional data prior to using it as input for auto-eventing. The filter used is [Savitzky-Golay](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter), and we need to specify the filtering polynomial (typically 1 or 2) and filtering window (an odd integer). For instance a window of 13 points on a 25 Hz signal is equivalent to using a window of half a second to smooth out the data. Based on our experience, we only need to filter ball position for Tracab 4 and SecondSpectrum. It is encouraged to filter the ball & player signal computing their x-y velocity components.

| Parameter | Definition |
| ------ | ------ |
| `filter_ball_speed` | Savitzky-Golay polynomial order and filtering window for ball speed, as a tuple of integers.|
| `filter_ball_position` | Savitzky-Golay polynomial order and filtering window for ball position, as a tuple of integers.|
| `filter_player_speed` | Savitzky-Golay polynomial order and filtering window for player speed, as a tuple of integers.|

### Team possession
Parameters for the `set_team_possession` routine.

| Parameter | Definition |
| ------ | ------ |
| `time_thresh` | Minimum time of possession by the same team consider possession has changed, in s.|
| `pass_thresh` | Minimum number of passes by the same team consider possession has changed.|
| `set_piece_thresh` | Minimum time between game resumes and a set piece is executed to identify team that has possession, in s.|

### Pressures
Parameters for the `set_pressures` routine. The last 4 parameters are not used in the current version of pressure found in `football_logic.py`, but rather on the test version in `pressures_test.py`.

| Parameter | Definition |
| ------ | ------ |
| `gk_z_limit` | Minimum and maximum ball height to consider a GK has the ball in hands, tuple of floats.|
| `angle_towards` | Angle to consider presser is going towards attacker, in degrees. |
| `moving_closer` | Lookbehind time to consider a player is moving closer to another one, in s.|
| `radius` | Max distance to consider a player is pressing another one from the front, in m.|
| `radius_back` | Max distance to consider a player is pressing another one from the back, in m.|
|`min_speed` | Minimum speed of player to consider is pressing if further than `radius_back` distance, in m/s.| 
|`gk_possession_lookahead` | Time interval to look ahead for GK ball possession, in s.| 
|`max_team_press_cutoff` | Maximum time for a continued team pressure, in s.| 
|`gk_z_threshold` | Maximum ball height beyond which there can be no pressure if the player is a goalkeeper, in m.| 
|`infielder_z_threshold` | Maximum ball height beyond which there can be no pressure if the player is an infielder, in m.| 
|`high_ball_z` & `high_ball_duration` | If the ball is above `high_ball_z` for longer than `high_ball_duration` between two consecutive pressures from the same team, no team pressure is considered during this interval. | 
                  
### Defensive modes
Parameters for the `set_defensive_modes` routine. The definitions for defensive modes are collected in the definitions-defensive-modes.pdf document in the repo.

| Parameter | Definition |
| ------ | ------ |
| `high_press_min_advanced_defenders` | Minimum number of defenders beyond black line in the document to consider a high press.|
| `mid_block_min_defenders_opp_half` | Minimum number of defenders in opposing half to consider a mid block.|
| `mid_block_pitch_fraction` | Fraction of pitch where mid-block is defined.|
| `low_block_max_defenders_opp_half` | Maximum number of defenders in opposing half to consider a low block. |
| `min_interval_duration` | Minimum duration of defensive mode to be considered valid and not collapsed, in s. |
| `transition_period_duration` | Maximum duration of transition period when possession changes, in s. |

### Pitch control
Parameters for the `apply_pitch_control` routine.

| Parameter | Definition |
| ------ | ------ |
| `vision_angle` | Players within -`vision_angle` to +`vision_angle` of the ball handler are considered to be within cone of vision, in degrees.|
| `pc_granularity` | Spatial resolution along a pass line where pitch control is evaluated, in m.|

## Output
The output of `event_detection` is a dictionary that contains the following items:
- `models`: Dictionary with several outputs:
    - `events`: A dataframe of all events & possession information automatically detected.
    - `db_intervals`: A dataframe with one row per dead ball interval, stating start/end as well as players satisfying the dead ball triggers and most likely set piece that resumed the play. Useful for debugging set pieces.
    - `team_possession`: A dataframe with one row per frame stating which team was in possession. 
    - `team_pressures`: A dataframe with one row per team pressure (optional). 
    - `defensive_modes`: A dataframe with one row per frame stating the defensive mode of the defending team. 
- `log_incidence`: A dictionary of all print statements that were issued, indexed by frame.
- `runtime`: A dictionary of the wall runtime for each routine in `set_event_detection`.
- `params`: The dictionary of parameters.

### Events
In this section we describe in more detail the columns that we find in the `events` dataframe. This dataframe has one row per frame of the tracking data where there is a ball-related action. Note that depending on the value of the boolean general parameters the `events` dataframe may not feature certain columns.

| Column | Definition |
| ------ | ------ |
| `frame` | Frame of the tracking data, integer.|
| `event_id` | Unique identifier for the event, string.|
| `event_time` | GameTime object from the `game_time.py` module.|
| `time_ms` | Time since the start of the half, in ms.|
| `ball_control` | Type of ball control, with library of values {no control, possession}, string.|
| `event_name` | Type of event, with library of values {pass, shot on target, shot not on target, own goal, cross, reception, interception, uncontrolled interception, save and retain, save and deflect, claim and retain, claim and deflect, reception from loose ball, unsuccessful save}, string. The event can be empty if for instance the action of the frame is the ball going out-of-play without any player in proximity.|
| `event_outcome` | Type of event outcome, with library of values {completed, unsuccessful, intercepted, intercepted without control, saved, goal, dead ball, end of half, deflected, claimed, loose ball, penalty missed}, string. The outcome can be empty.|
| `contest` | Type of contested possession, with library of values {contest, air duel, ground duel}, string. The contest can be empty.|
| `from_set_piece` | Type of set piece, with library of values {kickoff, penalty kick, free kick, corner kick, goal kick, throw-in, free kick?}, string. The set piece can be empty. The last tag corresponds to instance where the algorithm is confused.|
| `dead_ball_event` | Type of dead ball event, with library of values {goal, penalty awarded, foul, out for corner kick, out for goal kick, out for throw-in, foul?}, string. The dead ball event can be empty. The last tag corresponds to instance where the algorithm is confused.|
| `possession_outcome` | Type of possession outcome, with library of values {pass, maintained, lost, dead ball, goal, deflected}, string. The possession outcome can be empty.|
| `description` | Description of goals and penalties.|
| `home_score` | Home score at the end of the half.|
| `away_score` | Away score at the end of the half.|
| `assist_event_id` | Only for goals, ID of the event that constitutes the assist.|
| `previous_loss_event_id` | Only for possession gains, ID of the previous possession loss.|
| `next_gain_event_id` | Only for possession losses, ID of the next possession gain.|
| `next_loss_event_id` | Only for possession gains and losses, ID of the next possession loss.|
| `incoming_angle` | Only for possession gains, angle of the incoming pass, in degrees.|
| `outcoming_angle` | Only for possession gains and losses, angle of the outgoing pass, in degrees.|
| `player_location` | Only for possession gains and losses, location of the ball handler with respect to the opponent's convex hull with values {in front, inside, outside, flank, behind}, string.|
| `ball_from` | Only for possession gains, previous ball handler with values {teammate, opponent}, string.|
| `packing_possession` | Only for possession gains, amount of opponents overtaken by the possession that starts, integer.|
| `packing_pass` | Only for possession losses, amount of opponents overtaken, integer.|
| `distance_traveled_recipient` | Only for possession gains, distance traveled by recipient since prior possession loss, in m.|
| `distance_traveled_ball_x/y` | Only for possession losses, distance traveled by the ball until the next possession gain, in m.|
| `home/away_span_x/y` | Only for possession gains and losses, horizontal and vertical span of the home/away team, in m.|
| `ball_x/y/z` | Coordinates of the ball, in m. The origin (0,0,0) is at the center mark and the pitch is considered at z = 0.|
| `ball_speed_mps` | Speed of the ball, in m/s.|
| `is_gain` | Whether the event constitutes a possession gain, boolean.|
| `is_goalkeeper` | Whether the event involves any goalkeeper, boolean.|
| `ball_displacement` | Displacement of the ball from the current to the next frame, in m.|
| `change_in_direction` | Change in ball direction from the current to the next frame, as a cosine of the angle.|
| `change_in_speed` | Change in ball speed from the current to the next frame, in m/s.|
| `step_change` | Identifier to flag where in the algorithm was the event annotated, should be used for debugging purposes only, string.|
| `loss_gain` | Whether the raw possession model identified a possession loss or gain, should be used for debugging purposes only, string.|
| `team_possession` | Team ID in possession, string.|
| `player_id` | Player ID of the player performing the event, string. If multiple players are present the IDs are separated by ';' in ascending distance with respect to the ball. This condition applies to the below columns as well.|
| `x/y` | Coordinates of the player performing the event, string.|
| `team_id` | Team ID, string. |
| `distance` | Player distance to the ball, string (to accommodate multiple players). |
| `jersey_team` | Identifier of player using jersey number (team name).|
| `name` | Name of player.|
| `player_role` | Positional role of player.|
| `contest_player_id` | Player IDs in contest, string.|
| `contest_x/y` | Coordinates of the player in the contest, string. |
| `contest_team_id` | Team IDs of the players in contest, string. |
| `contest_distance` | Distance to the ball for contest players, string. |
| `contest_jersey_team` | Identifier of contest player using jersey number (team name).|
| `contest_name` | Name of contest players.|
| `contest_player_role` | Positional role of contest players.|
| `pressure` | Player IDs pressing the ball handler, string|
| `pressure_distance` | Distances from pressers to the ball handler, string|
| `defensive_mode` | Name of defensive mode selected.|
| `pc_gain` | Only for uncontested possession losses where team possession has been established, change in pitch control from location of teammates within cone of vision with respect to location of ball handler, dictionary.|
| `pc_gain` | Only for uncontested possession losses where team possession has been established, values of minimum-average-maximum pitch control along a passing line for teammates within cone of vision with respect to location of ball handler, dictionary of lists.|
| `within_vision` | Only for uncontested possession losses where team possession has been established, list of teammates within cone of vision of ball handler.|

       
# Video generation
A useful way to visualize and debug the generated events is the generation of tracking data videos with the events overlaid. The script to generate videos is `gen_video.py`, it can take several parameters that are well explained in the code and can be executed from the command line as follows:
```sh
python gen_video.py --dataset mwc_22 --provider tracab5 --game_id 128083 --half 1 --interval 0 1
```
The command above creates the video for the first two minutes (where `min` in [0,1]) of the first half of the game `128083` in the dataset denoted as `mwc_22`, which corresponds to the final.

Beware that this video creation tool is slow, namely to generate the video for a 45 minute half of a 25 Hz tracking data signal it takes approximately 90 minutes (double to the duration of the video).


## Python requirements
See the requirements.txt file in the repo.


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

