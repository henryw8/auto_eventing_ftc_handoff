#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:52:19 2020

Module for calculating a Pitch Control surface using MetricaSports's tracking & event data.

Pitch control (at a given location on the field) is the probability that a team will gain 
possession if the ball is moved to that location on the field. 

Methdology is described in "Off the ball scoring opportunities" by William Spearman:
http://www.sloansportsconference.com/wp-content/uploads/2018/02/2002.pdf

GitHub repo for this code can be found here:
https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking

Data can be found at: https://github.com/metrica-sports/sample-data

Functions
----------

calculate_pitch_control_at_target(): calculate the pitch control probability for the attacking and defending teams at a specified target position on the ball.

generate_pitch_control_for_passing_line(): calculate the pitch control probability for the passing lines between the ball's current position and the position of the
ball handler's teammates.

generate_pitch_control_for_event(): this function evaluates pitch control surface over the entire field at the moment
of the given event (determined by the index of the event passed as an input)

Classes
---------

The 'player' class collects and stores trajectory information for each player required by the pitch control calculations.

@author: Laurie Shaw (@EightyFivePoint)

"""


import numpy as np
import itertools

def default_model_params(time_to_control_veto=3, fps=25):
    """
    default_model_params()
    
    Returns the default parameters that define and evaluate the model. See Spearman 2018 for more details.
    
    Parameters
    -----------
    time_to_control_veto: If the probability that another team or player can get to the ball and control it is less than 10^-time_to_control_veto, ignore that player.
    
    
    Returns
    -----------
    
    params: dictionary of parameters required to determine and calculate the model
    
    """
    # key parameters for the model, as described in Spearman 2018
    params = {}
    # model parameters
    params['max_player_accel'] = 7. # maximum player acceleration m/s/s, not used in this implementation
    params['max_player_speed'] = 10. # maximum player speed m/s
    params['reaction_time'] = 0.7 # seconds, time taken for player to react and change trajectory. Roughly determined as vmax/amax
    params['tti_sigma'] = 0.45 # Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time
    params['kappa_def'] =  1. # kappa parameter in Spearman 2018 (=1.72 in the paper) that gives the advantage defending players to control ball, I have set to 1 so that home & away players have same ball control probability
    params['lambda_att'] = 4.3 # ball control parameter for attacking team
    params['lambda_def'] = 4.3 * params['kappa_def'] # ball control parameter for defending team
    params['lambda_gk'] = params['lambda_def']*3.0 # make goal keepers must quicker to control ball (because they can catch it)
    params['average_ball_speed'] = 15. # average ball travel speed in m/s
    # numerical parameters for model evaluation
    params['int_dt'] = 1/fps # integration timestep (dt)
    params['max_int_time'] = 10 # upper limit on integral time
    params['model_converge_tol'] = 0.01 # assume convergence when PPCF>0.99 at a given location.
    # The following are 'short-cut' parameters. We do not need to calculated PPCF explicitly when a player has a sufficient head start. 
    # A sufficient head start is when the a player arrives at the target location at least 'time_to_control' seconds before the next player
    params['time_to_control_att'] = time_to_control_veto*np.log(10) * (np.sqrt(3)*params['tti_sigma']/np.pi + 1/params['lambda_att'])
    params['time_to_control_def'] = time_to_control_veto*np.log(10) * (np.sqrt(3)*params['tti_sigma']/np.pi + 1/params['lambda_def'])
    return params

class player(object):
    """
    player() class
    
    Class defining a player object that stores position, velocity, time-to-intercept and pitch control contributions for a player
    
    __init__ Parameters
    -----------
    pid: id (jersey number) of player
    team: row of tracking data for team
    teamname: team name "Home" or "Away"
    params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
    

    methods include:
    -----------
    simple_time_to_intercept(r_final): time take for player to get to target position (r_final) given current position
    probability_intercept_ball(T): probability player will have controlled ball at time T given their expected time_to_intercept
    
    """
    # player object holds position, velocity, time-to-intercept and pitch control contributions for each player
    def __init__(self,pid,team,teamname,params,is_gk,pos_vel):
        self.id = pid
        self.is_gk = is_gk
        self.teamname = teamname
        self.playername = f"{teamname}_{pid}"
        self.vmax = params['max_player_speed'] # player max speed in m/s. Could be individualised
        self.reaction_time = params['reaction_time'] # player reaction time in 's'. Could be individualised
        self.tti_sigma = params['tti_sigma'] # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.lambda_att = params['lambda_att'] # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.lambda_def = params['lambda_gk'] if self.is_gk else params['lambda_def'] # factor of 3 ensures that anything near the GK is likely to be claimed by the GK
        self.position = np.array([np.nan, np.nan])
        self.velocity = np.array([np.nan, np.nan])
        self.all_pos_vel = pos_vel
        self.PPCF = 0. # initialise this for later
        
    def set_position(self,pos):
        self.position = np.array( [ pos[0], pos[1]] )
        
        
    def set_velocity(self,vel):
        self.velocity = np.array( [ vel[0], vel[1]] )
    
    def simple_time_to_intercept(self, r_final):
        self.PPCF = 0. # initialise this for later
        # Time to intercept assumes that the player continues moving at current velocity for 'reaction_time' seconds
        # and then runs at full speed to the target position.
        r_reaction = self.position + self.velocity*self.reaction_time
        self.time_to_intercept = self.reaction_time + np.linalg.norm(r_final-r_reaction)/self.vmax
        return self.time_to_intercept

    def probability_intercept_ball(self,T):
        # probability of a player arriving at target location at time 'T' given their expected time_to_intercept (time of arrival), as described in Spearman 2018
        f = 1/(1. + np.exp( -np.pi/np.sqrt(3.0)/self.tti_sigma * (T-self.time_to_intercept) ) )
        return f

    
def initialise_players(frame, players):
    """
    initialise_players(team,teamname,params)
    
    create a list of player objects that holds their positions and velocities from the tracking data dataframe 
    
    Parameters
    -----------
    
    team: row (i.e. instant) of either the home or away team tracking Dataframe
    teamname: team name "Home" or "Away"
    params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
        
    Returns
    -----------
    
    team_players: list of player objects for the team at at given instant
    
    """  
    players_frame = []
    for p in players:
        if frame in p.all_pos_vel:
            p.set_position(p.all_pos_vel[frame]['xy'])
            p.set_velocity(p.all_pos_vel[frame]['speed'])
            players_frame.append(p)
    return players_frame

def check_offsides( attacking_players, defending_players, ball_position, defending_half, verbose=False, tol=0.2):
    """
    check_offsides( attacking_players, defending_players, ball_position, verbose=False, tol=0.2):
    
    checks whetheer any of the attacking players are offside (allowing for a 'tol' margin of error). Offside players are removed from 
    the 'attacking_players' list and ignored in the pitch control calculation.
    
    Parameters
    -----------
        attacking_players: list of 'player' objects (see player class above) for the players on the attacking team (team in possession)
        defending_players: list of 'player' objects (see player class above) for the players on the defending team
        ball_position: Current position of the ball (start position for a pass). If set to NaN, function will assume that the ball is already at the target position.
        GK_numbers: tuple containing the player id of the goalkeepers for the (home team, away team)
        verbose: if True, print a message each time a player is found to be offside
        tol: A tolerance parameter that allows a player to be very marginally offside (up to 'tol' m) without being flagged offside. Default: 0.2m
            
    Returns
    -----------
        attacking_players: list of 'player' objects for the players on the attacking team with offside players removed
    """    
    # find the x-position of the second-deepest defending player (including GK)
    second_deepest_defender_x = sorted( [defending_half*p.position[0] for p in defending_players], reverse=True )[1]
    # define offside line as being the maximum of second_deepest_defender_x, ball position and half-way line
    offside_line = max(second_deepest_defender_x,defending_half*ball_position[0],0.0)+tol
    # any attacking players with x-position greater than the offside line are offside
    if verbose:
        for p in attacking_players:
            if p.position[0]*defending_half>offside_line:
                print("player %s in %s team is offside" % (p.id, p.playername) )
    attacking_players = [p for p in attacking_players if p.position[0]*defending_half<=offside_line]
    return attacking_players


def generate_pitch_control_for_event(players, event_frame, possession_team, active_goal, ball_start_pos, params, field_dimen = (105.,68.,), n_grid_cells_x = 50, offsides=True):
    """ generate_pitch_control_for_event
    
    Evaluates pitch control surface over the entire field at the moment of the given event (determined by the index of the event passed as an input)
    
    Parameters
    -----------
        event_id: Index (not row) of the event that describes the instant at which the pitch control surface should be calculated
        events: Dataframe containing the event data
        tracking_home: tracking DataFrame for the Home team
        tracking_away: tracking DataFrame for the Away team
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
        GK_numbers: tuple containing the player id of the goalkeepers for the (home team, away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        n_grid_cells_x: Number of pixels in the grid (in the x-direction) that covers the surface. Default is 50.
                        n_grid_cells_y will be calculated based on n_grid_cells_x and the field dimensions
        offsides: If True, find and remove offside atacking players from the calculation. Default is True.
        
    UPDATE (tutorial 4): Note new input arguments ('GK_numbers' and 'offsides')
        
    Returrns
    -----------
        PPCFa: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team.
               Surface for the defending team is just 1-PPCFa.
        xgrid: Positions of the pixels in the x-direction (field length)
        ygrid: Positions of the pixels in the y-direction (field width)

    """
    # get the details of the event (frame, team in possession, ball_start_position)
    pass_frame = event_frame
    ball_start_pos = ball_start_pos
    # break the pitch down into a grid
    n_grid_cells_y = int(n_grid_cells_x*field_dimen[1]/field_dimen[0])
    dx = field_dimen[0]/n_grid_cells_x
    dy = field_dimen[1]/n_grid_cells_y
    xgrid = np.arange(n_grid_cells_x)*dx - field_dimen[0]/2. + dx/2.
    ygrid = np.arange(n_grid_cells_y)*dy - field_dimen[1]/2. + dy/2.
    # initialise pitch control grids for attacking and defending teams 
    PPCFa = np.zeros( shape = (len(ygrid), len(xgrid)) )
    PPCFd = np.zeros( shape = (len(ygrid), len(xgrid)) )
    
    # initialise player positions and velocities for pitch control calc (so that we're not repeating this at each grid cell position)
    if possession_team=='home':
        attacking_players = initialise_players(event_frame, players['home'])
        defending_players = initialise_players(event_frame, players['away'])
    elif possession_team=='away':
        attacking_players = initialise_players(event_frame, players['away'])
        defending_players = initialise_players(event_frame, players['home'])
    else:
        assert False, "Team in possession must be either home or away"
    
    
    # find any attacking players that are offside and remove them from the pitch control calculation
    if offsides:
        attacking_players = check_offsides( attacking_players, defending_players, ball_start_pos, active_goal)
    # calculate pitch pitch control model at each location on the pitch
    for index in itertools.product(range(len(xgrid)), range(len(ygrid))):
        i,j = index[1],index[0]
        target_position = np.array( [xgrid[j], ygrid[i]] )
        PPCFa[i,j],PPCFd[i,j] = calculate_pitch_control_at_target(target_position, attacking_players, defending_players, ball_start_pos, params)
    # check probabilitiy sums within convergence
    checksum = np.sum( PPCFa + PPCFd ) / float(n_grid_cells_y*n_grid_cells_x ) 
    assert 1-checksum < params['model_converge_tol'], "Checksum failed: %1.3f" % (1-checksum)
    return PPCFa,xgrid,ygrid


def generate_pitch_control_for_passing_line(players, event_frame, possession_team, active_goal, ball_start_pos, 
                                            teammate_positions, params, granularity_m = 1, offsides=True):
    """ generate_pitch_control_for_passing_line
    
    Evaluates pitch control value over a given passing line at the moment of the given event
    
    Parameters
    -----------
        event_id: Index (not row) of the event that describes the instant at which the pitch control surface should be calculated
        events: Dataframe containing the event data
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
        n_grid_cells_x: Number of pixels in the grid (in the x-direction) that covers the surface. Default is 50.
                        n_grid_cells_y will be calculated based on n_grid_cells_x and the field dimensions
        offsides: If True, find and remove offside atacking players from the calculation. Default is True.
        
    UPDATE (tutorial 4): Note new input arguments ('GK_numbers' and 'offsides')
        
    Returns
    -----------
        results: List with one item per teammate, each item is a list with pitch control values and locations on the straight line from the
        current position of the ball to the position of the teammate

    """    
    # initialise player positions and velocities for pitch control calc (so that we're not repeating this at each grid cell position)
    if possession_team=='home':
        attacking_players = initialise_players(event_frame, players['home'])
        defending_players = initialise_players(event_frame, players['away'])
    elif possession_team=='away':
        attacking_players = initialise_players(event_frame, players['away'])
        defending_players = initialise_players(event_frame, players['home'])
    else:
        assert False, "Team in possession must be either home or away"
    
    ball_start_pos = np.array(ball_start_pos)
    # find any attacking players that are offside and remove them from the pitch control calculation
    if offsides:
        attacking_players = check_offsides( attacking_players, defending_players, ball_start_pos, active_goal)
    
    results = []
    PPCFa_0,PPCFd_0 = calculate_pitch_control_at_target(ball_start_pos, attacking_players, defending_players, ball_start_pos, params)
    
    for teammate in teammate_positions:
        
        # break the passing line into a mesh
        vector = np.array([teammate[0]-ball_start_pos[0], teammate[1]-ball_start_pos[1]])
        distance = np.linalg.norm(vector)
        norm_vector = vector/distance
        ds = max(int(np.ceil(distance/granularity_m)), 2)
        sgrid = distance*np.linspace(0,1,ds)
        
        # initialise pitch control grids for attacking and defending teams 
        PPCFa = np.zeros( shape = sgrid.shape )
        PPCFd = np.zeros( shape = sgrid.shape )
        PPCFa[0] = PPCFa_0
        PPCFd[0] = PPCFd_0
        # calculate pitch control model at each location on the pitch
        for i, ds_i in enumerate(sgrid[1:], 1):
            target_position = np.array( [ball_start_pos[0] + ds_i*norm_vector[0], ball_start_pos[1] + ds_i*norm_vector[1]] )
            PPCFa[i],PPCFd[i] = calculate_pitch_control_at_target(target_position, attacking_players, defending_players, ball_start_pos, params)
            
        # check probabilitiy sums within convergence
        checksum = np.sum( PPCFa + PPCFd ) / float(ds) 
        assert 1-checksum < params['model_converge_tol'], "Checksum failed: %1.3f" % (1-checksum)
        results.append([PPCFa,sgrid])
        
    return results

def calculate_pitch_control_at_target(target_position, attacking_players, defending_players, ball_start_pos, params):
    """ calculate_pitch_control_at_target
    
    Calculates the pitch control probability for the attacking and defending teams at a specified target position on the ball.
    
    Parameters
    -----------
        target_position: size 2 numpy array containing the (x,y) position of the position on the field to evaluate pitch control
        attacking_players: list of 'player' objects (see player class above) for the players on the attacking team (team in possession)
        defending_players: list of 'player' objects (see player class above) for the players on the defending team
        ball_start_pos: Current position of the ball (start position for a pass). If set to NaN, function will assume that the ball is already at the target position.
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
        
    Returrns
    -----------
        PPCFatt: Pitch control probability for the attacking team
        PPCFdef: Pitch control probability for the defending team ( 1-PPCFatt-PPCFdef <  params['model_converge_tol'] )

    """
    # calculate ball travel time from start position to end position.
    if ball_start_pos is None or any(np.isnan(ball_start_pos)): # assume that ball is already at location
        ball_travel_time = 0.0 
    else:
        # ball travel time is distance to target position from current ball position divided assumed average ball speed
        ball_travel_time = np.linalg.norm( target_position - ball_start_pos )/params['average_ball_speed']
    
    # first get arrival time of 'nearest' attacking player (nearest also dependent on current velocity)
    tau_min_att = np.nanmin( [p.simple_time_to_intercept(target_position) for p in attacking_players] )
    tau_min_def = np.nanmin( [p.simple_time_to_intercept(target_position ) for p in defending_players] )
    
    # check whether we actually need to solve equation 3
    if tau_min_att-max(ball_travel_time,tau_min_def) >= params['time_to_control_def']:
        # if defending team can arrive significantly before attacking team, no need to solve pitch control model
        return 0., 1.
    elif tau_min_def-max(ball_travel_time,tau_min_att) >= params['time_to_control_att']:
        # if attacking team can arrive significantly before defending team, no need to solve pitch control model
        return 1., 0.
    else: 
        # solve pitch control model by integrating equation 3 in Spearman et al.
        # first remove any player that is far (in time) from the target location
        attacking_players = [p for p in attacking_players if p.time_to_intercept-tau_min_att < params['time_to_control_att'] ]
        defending_players = [p for p in defending_players if p.time_to_intercept-tau_min_def < params['time_to_control_def'] ]
        # set up integration arrays
        dT_array = np.arange(ball_travel_time-params['int_dt'],ball_travel_time+params['max_int_time'],params['int_dt']) 
        PPCFatt = np.zeros_like( dT_array )
        PPCFdef = np.zeros_like( dT_array )
        # integration equation 3 of Spearman 2018 until convergence or tolerance limit hit (see 'params')
        ptot = 0.0
        i = 1
        while 1-ptot>params['model_converge_tol'] and i<dT_array.size: 
            T = dT_array[i]
            # PPCFatt[i] = params['int_dt']*sum([(1-PPCFatt[i-1]-PPCFdef[i-1])*p.probability_intercept_ball( T ) * p.lambda_att for p in attacking_players])
            # PPCFdef[i] = params['int_dt']*sum([(1-PPCFatt[i-1]-PPCFdef[i-1])*p.probability_intercept_ball( T ) * p.lambda_att for p in defending_players])
            # ptot1 = PPCFdef[i]+PPCFatt[i]
            for player in attacking_players:
                # calculate ball control probablity for 'player' in time interval T+dt
                dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * player.lambda_att
                # make sure it's greater than zero
                assert dPPCFdT>=0, 'Invalid attacking player probability (calculate_pitch_control_at_target)'
                player.PPCF += dPPCFdT*params['int_dt'] # total contribution from individual player
                PPCFatt[i] += player.PPCF # add to sum over players in the attacking team (remembering array element is zero at the start of each integration iteration)
            for player in defending_players:
                # calculate ball control probablity for 'player' in time interval T+dt
                dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * player.lambda_def
                # make sure it's greater than zero
                assert dPPCFdT>=0, 'Invalid defending player probability (calculate_pitch_control_at_target)'
                player.PPCF += dPPCFdT*params['int_dt'] # total contribution from individual player
                PPCFdef[i] += player.PPCF # add to sum over players in the defending team
            ptot = PPCFdef[i]+PPCFatt[i] # total pitch control probability 
            i += 1
        if i>=dT_array.size:
            print("Integration failed to converge: %1.3f" % (ptot) )
        return PPCFatt[i-1], PPCFdef[i-1]



    
