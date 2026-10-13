import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation, rc
# plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
import numpy as np
from game_time import GameTime, _display_football_time

#Allowing the animation to be displayed in a Notebook
# rc('animation', html='html5')

def _draw_football_field(l=105, w=68, color_dict={'pitch': 'w','lines': 'silver','ball': 'k'}, plot_aux=False, params=dict()):
    
    """
    This routine draws a football field with given length and width

    :param l: Length of pitch
    :param w: Width of pitch
    :param color_dict: Dictionary containing the default values to color the pitch
    :param plot_aux: Boolean to plot auxiliary lines for debugging
    :param params: Dictionary for tolerances to plot auxiliary lines        
    :return: Figure and axes
    """ 

    color_dict['aux'] = 'salmon'
    color_dict['tols'] = 'lightsteelblue'

    ## Setting up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots(figsize = (l/5, w/5))
    ax.set_xlim(( -3-l/2, l/2+3))
    ax.set_ylim((-4-w/2, w/2+10))
    
      
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ## Plotting the football Field
    ax.set_facecolor(color_dict['pitch'])#Grass
    ax.plot([0,0],[-w/2,w/2],color_dict['lines']) #half-way line
    ax.plot([-l/2,-l/2],[-w/2,w/2],color_dict['lines'])
    ax.plot([l/2,l/2],[-w/2,w/2],color_dict['lines'])
    ax.plot([-l/2,l/2],[-w/2,-w/2],color_dict['lines'])
    ax.plot([-l/2,l/2],[w/2,w/2],color_dict['lines'])

    ## Dimensions from Laws of the Game
    smbx_dim = 5.4864
    pnbx_len = 16.4592
    pnbx_wid = 40.2336
    pnspot_dim = 10.9728
    goal_dim = 7.3152
    arc_dim = 9.144
    y_circle = np.sin(53*np.pi/180)*arc_dim

    ax.add_artist(plt.Circle((0,0), arc_dim, edgecolor=color_dict['lines'],facecolor = None, fill=False)) #centre circle
    ax.add_artist(plt.Circle((0,0), 0.3, edgecolor=color_dict['lines'],facecolor = color_dict['lines'])) #centre spot

    ## Boxes

    ax.plot([pnbx_len-l/2,pnbx_len-l/2],[w/2-pnbx_len-goal_dim/2-w/2,w/2+pnbx_len+goal_dim/2-w/2],color_dict['lines'])
    ax.plot([l-pnbx_len-l/2,l-pnbx_len-l/2],[w/2-pnbx_len-goal_dim/2-w/2,w/2+pnbx_len+goal_dim/2-w/2],color_dict['lines'])
    ax.plot([smbx_dim-l/2,smbx_dim-l/2],[w/2-smbx_dim-goal_dim/2-w/2,w/2+smbx_dim+goal_dim/2-w/2],color_dict['lines'])
    ax.plot([l-smbx_dim-l/2,l-smbx_dim-l/2],[w/2-smbx_dim-goal_dim/2-w/2,w/2+smbx_dim+goal_dim/2-w/2],color_dict['lines'])
    ax.plot([0-l/2,pnbx_len-l/2],[w/2-pnbx_len-goal_dim/2-w/2,w/2-pnbx_len-goal_dim/2-w/2],color_dict['lines'])
    ax.plot([l-pnbx_len-l/2,l-l/2],[w/2-pnbx_len-goal_dim/2-w/2,w/2-pnbx_len-goal_dim/2-w/2],color_dict['lines'])
    ax.plot([0-l/2,pnbx_len-l/2],[w/2+pnbx_len+goal_dim/2-w/2,w/2+pnbx_len+goal_dim/2-w/2],color_dict['lines'])
    ax.plot([l-pnbx_len-l/2,l-l/2],[w/2+pnbx_len+goal_dim/2-w/2,w/2+pnbx_len+goal_dim/2-w/2],color_dict['lines'])
    ax.plot([0-l/2,smbx_dim-l/2],[w/2-smbx_dim-goal_dim/2-w/2,w/2-smbx_dim-goal_dim/2-w/2],color_dict['lines'])
    ax.plot([l-smbx_dim-l/2,l-l/2],[w/2-smbx_dim-goal_dim/2-w/2,w/2-smbx_dim-goal_dim/2-w/2],color_dict['lines'])
    ax.plot([0-l/2,smbx_dim-l/2],[w/2+smbx_dim+goal_dim/2-w/2,w/2+smbx_dim+goal_dim/2-w/2],color_dict['lines'])
    ax.plot([l-smbx_dim-l/2,l-l/2],[w/2+smbx_dim+goal_dim/2-w/2,w/2+smbx_dim+goal_dim/2-w/2],color_dict['lines'])
    ax.add_patch(mpl.patches.Arc((pnspot_dim-l/2,w/2-w/2),2*arc_dim,2*arc_dim,theta1=-53.0, theta2=53.0, edgecolor=color_dict['lines'],facecolor=color_dict['pitch']))
    ax.add_patch(mpl.patches.Arc((l-pnspot_dim-l/2,w/2-w/2),2*arc_dim,2*arc_dim,theta1=53.0+74, theta2=-53.0-74, edgecolor=color_dict['lines'],facecolor=color_dict['pitch']))
    ax.add_artist(plt.Circle((pnspot_dim-l/2,w/2-w/2), 0.4, edgecolor=color_dict['lines'],facecolor = color_dict['lines'])) #penalty spot
    ax.add_artist(plt.Circle((l-pnspot_dim-l/2,w/2-w/2), 0.4, edgecolor=color_dict['lines'],facecolor = color_dict['lines'])) #penalty spot

    ax.add_patch(mpl.patches.Arc((l-l/2,w-w/2),height=2,width=2,theta1=180,theta2=270, edgecolor=color_dict['lines'],facecolor=color_dict['pitch']))
    ax.add_patch(mpl.patches.Arc((l-l/2,0-w/2),height=2,width=2,theta1=90,theta2=180, edgecolor=color_dict['lines'],facecolor=color_dict['pitch']))
    ax.add_patch(mpl.patches.Arc((0-l/2,w-w/2),height=2,width=2,theta1=270,theta2=0, edgecolor=color_dict['lines'],facecolor=color_dict['pitch']))
    ax.add_patch(mpl.patches.Arc((0-l/2,0-w/2),height=2,width=2,theta1=0,theta2=90, edgecolor=color_dict['lines'],facecolor=color_dict['pitch']))

    ## Left Goal
    ax.plot([-2.44-l/2,-2.44-l/2],[w/2-goal_dim/2-w/2,w/2+goal_dim/2-w/2],color_dict['lines'])
    ax.plot([-2.44-l/2,0-l/2],[w/2-goal_dim/2-w/2,w/2-goal_dim/2-w/2],color_dict['lines'])
    ax.plot([-2.44-l/2,0-l/2],[w/2+goal_dim/2-w/2,w/2+goal_dim/2-w/2],color_dict['lines'])

    ## Right Goal
    ax.plot([l+2.44-l/2,l+2.44-l/2],[w/2-goal_dim/2-w/2,w/2+goal_dim/2-w/2],color_dict['lines'])
    ax.plot([l-l/2,l+2.44-l/2],[w/2-goal_dim/2-w/2,w/2-goal_dim/2-w/2],color_dict['lines'])
    ax.plot([l-l/2,l+2.44-l/2],[w/2+goal_dim/2-w/2,w/2+goal_dim/2-w/2],color_dict['lines'])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    
    ## Auxiliary lines for visualization and debugging purposes    
    if plot_aux:        
        ax.plot([l/2-arc_dim-l/2,l/2-arc_dim-l/2],[0-w/2,w-w/2],color_dict['aux'],ls='dashed', lw=1) #first third line
        ax.plot([l/2+arc_dim-l/2,l/2+arc_dim-l/2],[0-w/2,w-w/2],color_dict['aux'],ls='dashed', lw=1) #second third line
        ax.plot([0-l/2,l-l/2],[w/2+pnbx_wid/2-w/2,w/2+pnbx_wid/2-w/2],color_dict['aux'],ls='dashed', lw=1) #upper flank line
        ax.plot([0-l/2,l-l/2],[w/2-pnbx_wid/2-w/2,w/2-pnbx_wid/2-w/2],color_dict['aux'],ls='dashed', lw=1) #lower flank line

        corner_tol = params.get('tol_ck',2)
        ko1_tol = params.get('tol_ko1',3)
        ko2_tol = params.get('tol_ko2',2)
        pk1_tol = params.get('tol_pk1',1)
        pk2_tol = params.get('tol_pk2',2)
        pk3_tol = params.get('tol_pk3',2)
        ti_tol = params.get('tol_ti',1)
        gk_tol = params.get('tol_gk',1)

        ax.plot([l/2 - ko1_tol-l/2,l/2 - ko1_tol-l/2],[0-w/2,w-w/2],color_dict['tols'], ls='dashed', lw=1) #kickoff lines
        ax.plot([l/2 + ko1_tol-l/2,l/2 + ko1_tol-l/2],[0-w/2,w-w/2],color_dict['tols'], ls='dashed', lw=1) #
        ax.add_artist(plt.Circle((l/2-l/2,w/2-w/2), ko2_tol, edgecolor=color_dict['tols'],facecolor = None, fill=False, ls='dashed',lw=1)) #centre circle

        ax.plot([0-l/2,l-l/2],[w-ti_tol-w/2,w-ti_tol-w/2],color_dict['tols'],ls='dashed', lw=1) #throwin lines
        ax.plot([0-l/2,l-l/2],[ti_tol-w/2,ti_tol-w/2],color_dict['tols'],ls='dashed', lw=1) #

        ax.add_patch(mpl.patches.Arc((l-l/2,w-w/2),height=2*corner_tol,width=2*corner_tol,theta1=180,theta2=270, ls='dashed', lw=1, edgecolor=color_dict['tols'], fill=False))
        ax.add_patch(mpl.patches.Arc((l-l/2,0-w/2),height=2*corner_tol,width=2*corner_tol,theta1=90,theta2=180, ls='dashed', lw=1, edgecolor=color_dict['tols'], fill=False))
        ax.add_patch(mpl.patches.Arc((0-l/2,w-w/2),height=2*corner_tol,width=2*corner_tol,theta1=270,theta2=0, ls='dashed', lw=1, edgecolor=color_dict['tols'], fill=False))
        ax.add_patch(mpl.patches.Arc((0-l/2,0-w/2),height=2*corner_tol,width=2*corner_tol,theta1=0,theta2=90, ls='dashed', lw=1, edgecolor=color_dict['tols'], fill=False))
        ax.plot([0-l/2,smbx_dim+gk_tol-l/2],[w/2-smbx_dim-goal_dim/2-gk_tol-w/2,w/2-smbx_dim-goal_dim/2-gk_tol-w/2],color_dict['tols'], ls='dashed', lw=1)
        ax.plot([l-smbx_dim-gk_tol-l/2,l-l/2],[w/2-smbx_dim-goal_dim/2-gk_tol-w/2,w/2-smbx_dim-goal_dim/2-gk_tol-w/2],color_dict['tols'], ls='dashed', lw=1)
        ax.plot([0-l/2,smbx_dim+gk_tol-l/2],[w/2+smbx_dim+goal_dim/2+gk_tol-w/2,w/2+smbx_dim+goal_dim/2+gk_tol-w/2],color_dict['tols'], ls='dashed', lw=1)
        ax.plot([l-smbx_dim-gk_tol-l/2,l-l/2],[w/2+smbx_dim+goal_dim/2+gk_tol-w/2,w/2+smbx_dim+goal_dim/2+gk_tol-w/2],color_dict['tols'], ls='dashed', lw=1)
        ax.plot([smbx_dim+gk_tol-l/2,smbx_dim+gk_tol-l/2],[w/2-smbx_dim-goal_dim/2-gk_tol-w/2,w/2+smbx_dim+goal_dim/2+gk_tol-w/2],color_dict['tols'], ls='dashed', lw=1)
        ax.plot([l-smbx_dim-gk_tol-l/2,l-smbx_dim-gk_tol-l/2],[w/2-smbx_dim-goal_dim/2-gk_tol-w/2,w/2+smbx_dim+goal_dim/2+gk_tol-w/2],color_dict['tols'], ls='dashed', lw=1)

        ax.add_patch(mpl.patches.Rectangle((l-pnspot_dim-pk2_tol-l/2,w/2-pk2_tol-w/2),height=2*pk2_tol,width=1.5*pk2_tol, ls='dashed', lw=1, edgecolor=color_dict['tols'], fill=False))
        ax.add_patch(mpl.patches.Rectangle((0 + pnspot_dim-0.5*pk2_tol-l/2,w/2-pk2_tol-w/2),height=2*pk2_tol,width=1.5*pk2_tol, ls='dashed', lw=1, edgecolor=color_dict['tols'], fill=False))
        ax.add_patch(mpl.patches.Rectangle((l-pk1_tol-l/2,w/2-goal_dim/2-pk1_tol-w/2),height=2*pk1_tol+goal_dim,width=pk1_tol, ls='dashed', lw=1, edgecolor=color_dict['tols'], fill=False))
        ax.add_patch(mpl.patches.Rectangle((0-l/2,w/2-goal_dim/2-pk1_tol-w/2),height=2*pk1_tol+goal_dim,width=pk1_tol, ls='dashed', lw=1, edgecolor=color_dict['tols'], fill=False))

        ax.plot([pnbx_len-pk3_tol-l/2,pnbx_len-pk3_tol-l/2],[w/2-pnbx_len-goal_dim/2+pk3_tol-w/2,w/2-y_circle-w/2],color_dict['tols'], ls='dashed', lw=1)
        ax.plot([pnbx_len-pk3_tol-l/2,pnbx_len-pk3_tol-l/2],[w/2+y_circle-w/2,w/2+pnbx_len+goal_dim/2-pk3_tol-w/2],color_dict['tols'], ls='dashed', lw=1)
        ax.plot([l-pnbx_len+pk3_tol-l/2,l-pnbx_len+pk3_tol-l/2],[w/2-pnbx_len-goal_dim/2+pk3_tol-w/2,w/2-y_circle-w/2],color_dict['tols'], ls='dashed', lw=1)
        ax.plot([l-pnbx_len+pk3_tol-l/2,l-pnbx_len+pk3_tol-l/2],[w/2+y_circle-w/2,w/2+pnbx_len+goal_dim/2-pk3_tol-w/2],color_dict['tols'], ls='dashed', lw=1)
        ax.plot([0-l/2,pnbx_len-pk3_tol-l/2],[w/2-pnbx_len-goal_dim/2+pk3_tol-w/2,w/2-pnbx_len-goal_dim/2+pk3_tol-w/2],color_dict['tols'], ls='dashed', lw=1)
        ax.plot([l-pnbx_len+pk3_tol-l/2,l-l/2],[w/2-pnbx_len-goal_dim/2+pk3_tol-w/2,w/2-pnbx_len-goal_dim/2+pk3_tol-w/2],color_dict['tols'], ls='dashed', lw=1)
        ax.plot([0-l/2,pnbx_len-pk3_tol-l/2],[w/2+pnbx_len+goal_dim/2-pk3_tol-w/2,w/2+pnbx_len+goal_dim/2-pk3_tol-w/2],color_dict['tols'], ls='dashed', lw=1)
        ax.plot([l-pnbx_len+pk3_tol-l/2,l-l/2],[w/2+pnbx_len+goal_dim/2-pk3_tol-w/2,w/2+pnbx_len+goal_dim/2-pk3_tol-w/2],color_dict['tols'], ls='dashed', lw=1)
        ax.add_patch(mpl.patches.Arc((pnspot_dim-pk3_tol-l/2,w/2-w/2),2*arc_dim,2*arc_dim,theta1=-53.0, theta2=53.0, edgecolor=color_dict['tols'],facecolor=color_dict['pitch'], ls='dashed', lw=1))
        ax.add_patch(mpl.patches.Arc((l-pnspot_dim+pk3_tol-l/2,w/2-w/2),2*arc_dim,2*arc_dim,theta1=53.0+74, theta2=-53.0-74, edgecolor=color_dict['tols'],facecolor=color_dict['pitch'], ls='dashed', lw=1))


    return fig,ax

class WholePitchVideo():
  def __init__(self, tracking_data, players_df, frames_to_skip=0, speed=1,
               events=pd.DataFrame(), green_bckg=False, plot_pz=False, plot_direction=False, 
               team_colors=None, n=3, plot_aux=False, params=dict(), 
               display_time='gametime', team_left_right=None,
               use_index='frame', flip_xy=False,
               team_stat=pd.DataFrame(),
               additional_ball=None, time_from_zero=True, plot_refs=True, 
               annotate_tag=False, verbose=10, fps=25, highlight_visible=False,
               player_display='jersey_team',
              plot_circles=True):

    """
    Creates an animation class that allows the user to generate a 
    video with the top-view of a game, for a particular time range

    :param tracking_data: Dataframe of tracking data
    :param players_df: Dataframe of player information
    :param frames_to_skip: If positive the video will skip the number of frames stated by the variable
    :param speed: If 1, video will be shown at regular speed
    :param events: Dictionary for events to plot
    :param green_bckg: Boolean to plot the background green
    :param plot_pz: Boolean to plot the possession zone of each player
    :param plot_direction: Boolean to plot the direction of each player
    :param team_colors: Dictionary to modify colors of teams
    :param n=3: Use n previous position values to represent speed
    :param plot_aux: Boolean to plot auxiliary pitch lines for debugging
    :param params: Dictionary containing relevant parameters
    :params display_time: Time information to show
    :params team_left_right: Dictionary to set which team is playing on the left and which on the right. If none, it will be determined automatically
    :params use_index: Variable to use for indexing the distinct time instants
    :params flip_xy: Boolean to flip the x-y coordinates of all players and ball,
    :params team_stat: Dataframe to represent a given team stat, for instance defensive modes or team possession
    :params additional_ball: Dataframe with positional values for an additional ball
    :params time_from_zero: Time starts from 0 at the beginning of each half. Only important if display_time is set to 'gametime
    :params plot_refs: Boolean to plot referees
    :params annotate_tag: Column on the tracking data to use for identifying players on the video
    :params verbose: Integer to show at what percentages of video completion will a print statement appear. Default 10 means every 10%
    :params fps: Sampling fps in fps
    :params highlight_visible: If the 'player_visible' column is on the tracking data, setting this to True will color visible/invisible players differently
    :params player_display: Column on the tracking data to use for identifying events by players
    :params plot_circles: Boolean to plot circles for passes/shots/crosses/duels to be able to identify them better for visualization purposes       
    :return: Video animation
    """ 
    if not plot_refs:
        tracking_data = tracking_data.query("team_id != 'referee'")

    tracking_data.reset_index(drop=True, inplace=True)

    ## Establish some key variables to be used throughout the routine
    self.input = tracking_data
    self.players_df = players_df.set_index('tracking_player_id')
    self.tracking_pitch_length = params['pitch_length']
    self.tracking_pitch_width = params['pitch_width']
    self.players = tracking_data['player_id'].unique().tolist()
    self.annotate_tag = annotate_tag
    self.verbose = verbose
    if self.annotate_tag:
        tm = tracking_data.drop_duplicates(subset=['player_id'])[['player_id', annotate_tag]].copy()
        self.players_tag = dict(zip(tm['player_id'], tm[annotate_tag]))
    else:
        self.players_tag = {}
    
    suff_home = ' (blue)' if team_colors is None or 'blue' in team_colors['home'] else f" ({team_colors['home']})"
    suff_away = ' (orange)' if team_colors is None or 'red' in team_colors['away'] or 'orange' in team_colors['away'] else f" ({team_colors['away']})"
    suff_home = ''
    suff_away = ''
    self.home_name = players_df.iloc[0]['home_team_name']
    self.away_name = players_df.iloc[0]['away_team_name']
    self.home_tag = players_df.iloc[0]['home_team_name'] + suff_home 
    self.away_tag = players_df.iloc[0]['away_team_name'] + suff_away 
    self.home_id = players_df.iloc[0]['tracking_home_id']
    self.away_id = players_df.iloc[0]['tracking_away_id']
    self.goalkeepers = list(player for player in self.players if player in players_df.query("is_goalkeeper == 1")['tracking_player_id'].tolist())
    
    self.params = params
    self.plot_circles = plot_circles

    self.n = n # sense of speed
    self.n_player = 2
    
    if flip_xy:
        self.flip_val = -1
    else:
        self.flip_val = 1
        
    if plot_pz:
        self.player_size = 8
        self.ball_size = 4
    else:
        self.player_size = 10
        self.ball_size = 5

    ## Color of teams    
    self.color_map = {'referee': 'indigo',
                 self.home_name: 'royalblue' if team_colors is None else team_colors.get('home','royalblue'),
                 self.away_name: 'orangered' if team_colors is None else team_colors.get('away','orangered'),
                 'goalkeepers' : 'mediumseagreen' if team_colors is None else team_colors.get('goalkeepers','mediumseagreen'),
                 'invisible': 'lightgrey'
    }
    
    ## Color of events
    possession_color = 'goldenrod'
    
    self.color_events = {'possession': possession_color,
                         'reception':possession_color,
                         'interception':possession_color,
                         'cross': 'olive',
                         'pass':  possession_color,
                         'shot on target':'limegreen',
                         'shot not on target': 'darkgreen',
                         'own goal':'turquoise',
                         'shot on target?':'lightgreen',
                         'reception from loose ball':'paleturquoise',
                         'save and deflect':'magenta',
                         'save and retain':'magenta',
                         'claim and retain':'blueviolet',
                         'claim and deflect':'blueviolet',
                         'unsuccessful save':'midnightblue',
                         'ground duel':'saddlebrown',
                         'air duel':'dodgerblue',
                         'contest':'black',
                         'tackle':'black'}
    
    ## Color of pitch
    if green_bckg == True:
      color_pitch = {'pitch': 'g',
                 'lines': 'w',
                 'ball': 'w',
                 'ball_add':'salmon'
      }
      
    elif green_bckg == False:
      color_pitch = {'pitch': 'w',
                 'lines': 'silver',
                 'ball': 'k',
                 'ball_add': 'tan'
      }

    self.home_away_map = {
            self.home_id: self.home_name,
            self.away_id: self.away_name,
            'referee': 'referee'
        }
    
    self.acronym_map = {
            self.home_id: players_df.iloc[0]['home_acronym'],
            self.away_id: players_df.iloc[0]['away_acronym'],
            'referee': 'ref'
        }
    
    
    self.team_dict = {player_id: self.players_df.at[player_id,'team_name'] for player_id in self.players}

    self.color_dict = {player_id: (self.color_map[team] if player_id not in self.goalkeepers else self.color_map['goalkeepers']) for (player_id, team) in self.team_dict.items()} 
    self.color_pitch = color_pitch
    self.color_cz = 'khaki'
    self.plot_pz = plot_pz
    self.direction_color = 'black'
    self.plot_direction = plot_direction
    self.direction_length = 1/2 #7.2km/h is a length of 1

    unique_data = tracking_data.drop_duplicates(subset=use_index) 
    self.frames = unique_data[use_index].tolist()

    ## Getting data from the players
    self.xy_data = {}
    self.xy_direction = {}
    self.player_visible = {}

    for player in self.players:
      xy_data = pd.DataFrame(index=self.frames,columns=['x','y'])
      track_player = tracking_data.loc[tracking_data['player_id']==player,['x','y',use_index]].copy().set_index(use_index)
      xy_data.loc[track_player.index] = track_player
      self.xy_data[player] = self.flip_val*xy_data.to_numpy(dtype=np.float64)  

      if highlight_visible and 'player_visible' in unique_data.columns:
        player_visible = pd.DataFrame(index=self.frames,columns=['is_visible'])
        visible_bools = tracking_data.loc[tracking_data['player_id']==player,['player_visible',use_index]].copy().set_index(use_index)
        player_visible.loc[visible_bools.index] = visible_bools
        self.player_visible[player] = player_visible['is_visible'].tolist()  
      else:
        self.player_visible[player] = self.xy_data[player].shape[0]*[True]

      if plot_direction:
        if 'vx' in tracking_data.columns:
            xy_dir = pd.DataFrame(index=self.frames,columns=['vx','vy'])
            dir_player = tracking_data.loc[tracking_data['player_id']==player,['vx','vy',use_index]].copy().set_index(use_index)
            xy_dir.loc[dir_player.index] = dir_player
            self.xy_direction[player] = list(zip(self.flip_val*xy_dir.vx, self.flip_val*xy_dir.vy))
        else:
            track_player['diff_x'] = track_player.x.diff()
            track_player['diff_y'] = track_player.y.diff()
            track_player['direction_x'] = np.where((track_player['diff_x'] == 0) & (track_player['diff_y'] == 0), 0, np.round(track_player['diff_x']/np.sqrt(track_player['diff_x']**2+track_player['diff_y']**2),4)) 
            track_player['direction_y'] = np.where((track_player['diff_x'] == 0) & (track_player['diff_y'] == 0), 0, np.round(track_player['diff_y']/np.sqrt(track_player['diff_x']**2+track_player['diff_y']**2),4)) 
            self.xy_direction[player] = list(zip(self.flip_val*track_player.direction_x, self.flip_val*track_player.direction_y))

    ## Setting time annotation
    if display_time == 'gametime':
        self.display_time = unique_data.apply(lambda x: _display_football_time(int(x['half']),int(x['minute']),x['sec'],time_from_zero), axis=1).tolist()
    elif display_time == 'timestamp':
        self.display_time = unique_data['collected_at'].tolist()
    else:
        print("No time mode specified")
        self.display_time = len(unique_data['frame'].tolist())*['']
    
    ## Setting ball position, status, visibility and additional ball (optional)    
    self.xy_ball = self.flip_val*unique_data[['ball_x','ball_y']].to_numpy(dtype=np.float64)
    if 'ball_status' not in unique_data.columns:
        self.ball_status = unique_data.shape[0]*[None]
    else:
        self.ball_status = unique_data['ball_status'].tolist()
    
    if highlight_visible and 'ball_visible' in unique_data.columns:
        self.ball_visible = unique_data['ball_visible'].tolist()
    else:
        self.ball_visible = unique_data.shape[0]*[True]
    
    if additional_ball is not None:
        self.xy_ball_add = self.flip_val*additional_ball[['ball_x','ball_y']].to_numpy(dtype=np.float64)
    else:
        self.xy_ball_add = None

    ## Setting team stat (such as possession or defensive mode) to be shown as an annotation
    if not team_stat.empty:
        if 'defensive_mode' in team_stat.columns.tolist():
            team_stat['display'] = team_stat.apply(lambda x: x['defensive_mode'].upper() if x['defensive_mode_context'] != 'UNCLEAR' else x['defensive_mode'].upper() + ' (filled)', axis=1)
            self.display_team_stat_text = team_stat.apply(lambda x: x['display']+ ' by ' + 
                                     self.acronym_map[str(x['defending_team_id'])] if x['ball_status'] == 'in_play' else 'DEAD BALL', axis=1).tolist()
            self.display_team_stat_color = team_stat.apply(lambda x: team_colors['home' if self.home_id == x['defending_team_id'] else 'away'], axis=1).tolist()
            
        elif 'offensive_mode' in team_stat.columns.tolist():
            self.display_team_stat_text = team_stat.apply(lambda x: x['offensive_mode'].upper()+ ' by ' +
                                     self.acronym_map[str(x['attacking_team_id'])] if x['ball_status'] == 'in_play' else 'DEAD BALL', axis=1).tolist()
            
            self.display_team_stat_color = team_stat.apply(lambda x: team_colors['home' if self.home_id == x['attacking_team_id'] else 'away'], axis=1).tolist()
                
        elif 'attacking_team_id' in team_stat.columns.tolist():
            
            self.display_team_stat_text = ['POSSESSION '+ self.acronym_map[str(s)] for s in team_stat['attacking_team_id'].tolist()]
            self.display_team_stat_color = team_stat['attacking_team_id'].apply(lambda x: team_colors['home' if self.home_id == x else 'away']).tolist()                  
        else:
            print('No team stat found')
    else:
        self.display_team_stat_text = None
        self.display_team_stat_color = None
        
    # Setting which team is on which pitch
    if team_left_right is None:
        if len(self.goalkeepers) == 2:
            team_left, team_right = self._get_team_field_goalies()
        else:
            print(f'{len(self.goalkeepers)} GOALIES WERE FOUND, USING ALL PLAYERS')
            team_left, team_right = self._get_team_field_players()
    else:
        team_left = self.home_away_map[team_left_right[0]]
        team_right = self.home_away_map[team_left_right[1]]


    if team_left == self.home_name:
        home_first = True
    elif team_right == self.home_name:
        home_first = False
    else:
        print("WEIRD, should check team names correspondence")

    ## Set event & score annotations, as well as frames where the shot/pass/save circle will linger
    self.shot_fade = 50
    self.shot_counter = self.shot_fade
    self.shooting_player = ''

    self.pass_fade = 25
    self.pass_counter = self.pass_fade
    self.passing_player = ''

    self.save_fade = 50
    self.save_counter = self.save_fade
    self.saving_player = ''

    self.plot_all_events = False
    self.annotation_fade = 50

    if not events.empty:
      self.plot_all_events = True
      events['frame'] = events['frame'].astype('int64')
      frames_events = events['frame'].tolist()
      events = events.set_index('frame')

      self.event_annotation = []
      self.event_name = []
      self.control_name = []
      self.plot_this_event = []
      self.event_player_id = []
      self.display_score = []
      self.event_color = []
      self.control_color = []
      self.contest_name = []
        
      if 'home_score' in events.columns:
        if home_first:
          home_score = events.iloc[0]['home_score']
          away_score = events.iloc[0]['away_score']
        else:
          home_score = events.iloc[0]['away_score']
          away_score = events.iloc[0]['home_score']

      if 'dead_ball_event' in events.columns:
        dead_ball_event = events.iloc[0]['dead_ball_event']

      ct = 1
      for i, frame in enumerate(self.frames):
        control_color = dict()
        event_color = ''
        event_annotation = ''
        if ct > self.annotation_fade or i == 0:
          annotation_linger = ''

        if np.isnan(self.xy_ball[i,:]).any():
          event_annotation = ''

        if frame in frames_events:
          control_name = events.at[frame,'ball_control']
          contest_name = events.at[frame,'contest']
          event_name = events.at[frame,'event_name']
          player_id = events.at[frame,'player_id'].split(';')[0]

          if 'from_set_piece' in events.columns:
            from_set_piece = events.at[frame,'from_set_piece']
          else:
            from_set_piece = ''
          
          if control_name != 'no control' and (event_name in list(self.color_events.keys()) or control_name in list(self.color_events.keys()) or 'duel' in contest_name):            
            player_annotation = events.at[frame,player_display].replace(';','-')
            contest_player_annotation = events.at[frame,'contest_'+player_display].replace(';','-')
            annotation_linger = ''

            if any(substring in event_name for substring in ['shot','pass','cross','save','claim','loose','own']):

              # event_color[player_id] = self.color_events[event_name]
              event_color = self.color_events[event_name]

              if 'duel' in contest_name:
                event_annotation = f'{event_name} from {contest_name}'.upper() + f" by {contest_player_annotation}"
                event_color = self.color_events[contest_name]
              else:
                if from_set_piece != '':
                  event_annotation = event_name.upper() + ' from' + ' '+ from_set_piece.upper() + f" by {player_annotation}"
                else:
                  event_annotation = event_name.upper() + f" by {player_annotation}"

              annotation_linger = event_annotation
              ct = 1
            else:
              
              if any(substring in control_name for substring in ['possession']):
                event_annotation = control_name + f" by {player_annotation}"
                
              if control_name == 'possession':
                control_color[player_id] = self.color_events['possession']
          else:
            if annotation_linger != '':
                event_annotation = annotation_linger
            
          if 'dead_ball_event' in events.columns:
                         
            if event_name != '' or events.at[frame,'dead_ball_event']!= '':
              dead_ball_event = events.at[frame,'dead_ball_event'].upper()

            if dead_ball_event == 'GOAL':
              last_shot = events.loc[:frame,'event_name'].str.contains('|'.join(['shot','own']))[::-1].idxmax()
              db_annotation = events.at[last_shot,player_display].split(';')[0]
              if 'shot' in events.at[last_shot,'event_name']: 
                dead_ball_event = f"GOAL by {db_annotation}"
              else:
                dead_ball_event = f"OWN GOAL by {db_annotation}"
            
            if event_annotation == '':
              event_annotation = dead_ball_event
          else:
            dead_ball_event = ''         

          self.event_color.append(event_color)
          self.control_color.append(control_color)
          self.event_annotation.append(event_annotation)
          self.control_name.append(control_name)
          self.contest_name.append(contest_name)
          self.event_name.append(event_name)
          self.plot_this_event.append(event_name in list(self.color_events.keys()) or control_name in list(self.color_events.keys()) or 'duel' in contest_name)
          self.event_player_id.append(player_id)
          
          if 'home_score' in events.columns:
            home_score = events.at[frame,'home_score']
            away_score = events.at[frame,'away_score']
            if home_first:
              self.display_score.append(f"{home_score}:{away_score}")
            else:
              self.display_score.append(f"{away_score}:{home_score}")

          else:
            self.display_score.append('')

        else:
          ct += 1
          self.event_color.append(event_color)
          self.control_color.append(control_color)
          self.control_name.append('')
          self.contest_name.append('')
          self.event_name.append('')     
          self.plot_this_event.append(False)   
          self.event_player_id.append('') 
          if dead_ball_event != '':
            self.event_annotation.append(dead_ball_event)
          else:
            if annotation_linger != '' and ct <= self.annotation_fade and event_annotation == '':
              self.event_annotation.append(annotation_linger)
            else:
              self.event_annotation.append('')

          if 'home_score' in events.columns:
            if home_first:
              self.display_score.append(f"{home_score}:{away_score}")
            else:
              self.display_score.append(f"{away_score}:{home_score}")

          else:
            self.display_score.append('')
    

    ## Plotting the football field
    self.fig, self.ax = _draw_football_field(self.tracking_pitch_length,
                        self.tracking_pitch_width,self.color_pitch,
                        plot_aux=plot_aux, params=params)

    ## Creating the points for displaying the players (with direction optional) and the ball
    self.point_list = self._create_point_list()   
    self.annot_list = self._create_annot_list()

    if plot_pz:
        self.pz_list = self._create_pz_list()
    else:
        self.pz_list = len(self.players)*[None]
    
    if plot_direction:
        self.direction_list = self._create_direction_list()
    else:
        self.direction_list = [[None] for player in self.players]


    self.player_info = [(self.point_list[j][0], self.pz_list[j], self.direction_list[j][0], self.xy_data[player], self.player_visible[player], self.xy_direction.get(player,None), self.color_dict[player], self.annot_list[j], player) for j, player in enumerate(self.players)]    
    self.point_ball = self.ax.plot([],[],lw=self.ball_size)
    self.point_ball_add = self.ax.plot([],[],lw=self.ball_size)

    ## Create the points for the lingering visualization of discrete events
    self.point_duel = self.ax.plot([],[],lw=2)
    self.point_pass = self.ax.plot([],[],lw=2)
    self.point_shot = self.ax.plot([],[],lw=2)
    self.point_save = self.ax.plot([],[],lw=2)


    ## Creating the different annotation objects, some of them static, some of them dynamic
    self.ax.annotate(text=self.home_tag if team_left == self.home_name else self.away_tag,
                   xy=(-2-self.tracking_pitch_length/2,
                     self.tracking_pitch_width/2+2),
                   fontsize = 18,
                   ha='left')

    self.ax.annotate(text=self.home_tag if team_right == self.home_name else self.away_tag,
                   xy=(2+self.tracking_pitch_length/2,
                     self.tracking_pitch_width/2+2),
                   fontsize = 18,
                   ha='right')  

    self.info_time = self.ax.annotate(text='',
                            xy=(0,
                              self.tracking_pitch_width/2+8.5),
                            fontsize = 22,
                            ha='center')
    
    self.info_score = self.ax.annotate(text='',
                            xy=(0,
                              self.tracking_pitch_width/2+6),
                            fontsize = 20,
                            ha='center')
    
    if self.display_team_stat_text is None:
        self.info_event = self.ax.annotate(text='',
                                 xy=(0,
                                   self.tracking_pitch_width/2+3),
                                 fontsize = 18,
                                 ha='center')
    else:
        
        self.info_team_stat = self.ax.annotate(text='',
                                xy=(0,
                                  self.tracking_pitch_width/2+3.5),
                                fontsize = 18,
                                ha='center')

        self.info_event = self.ax.annotate(text='',
                                 xy=(0,
                                   self.tracking_pitch_width/2+1),
                                 fontsize = 18,
                                 ha='center')
    self.progress = [0]
    #Call the animator

    self.anim = animation.FuncAnimation(self.fig,
                                   self._animate,
                                   init_func= self._init,
                                   frames=range(0, len(self.display_time), frames_to_skip+1),
                                   interval=1000/(fps*speed),
                                   blit=True,
                                  )
    plt.close() #allows to not plot the animation until is called

  def get_anim(self):
    ## Returns the actual animation as a video object to be displayed within a Jupyter notebook or saved to a file
    return self.anim
    
  def _create_point_list(self):
    ## Creates the list of player annotations without coordinates specified
    point_list = []
    for player in self.players:
      point_list.append(self.ax.plot([],[],lw=self.player_size))
    return point_list

  def _create_annot_list(self):
    ## Creates the list of text annotations without text specified
    annot_list = []
    for player in self.players:
      annot_list.append(self.ax.text(x=np.nan, y=np.nan, s='',fontsize=13))
    return annot_list 

  def _create_direction_list(self):
    ## Creates the list of player directions without coordinates specified
    point_list = []
    for player in self.players:
      point_list.append(self.ax.plot([],[],lw=1, color=self.direction_color))
    return point_list


  def _create_pz_list(self):
    ## Creates the list of player PZ without coordinates specified
    point_list = []
    for player in self.players:
      c = plt.Circle((0, 0), radius=1,facecolor=self.color_cz, edgecolor='black',
                    alpha=0.8, visible=True, lw=0.5)
      point = self.ax.add_patch(c)
      point_list.append(point)
    return point_list

  def _get_team_field_goalies(self):
    ## Get the team playing to the left/right using the goalkeepers
    team_left = ''
    team_right = ''
    for goalkeeper in self.goalkeepers:
        if np.nanmean(self.xy_data[goalkeeper][:,0]) < 0:
            team_left = self.team_dict[goalkeeper]
        elif np.nanmean(self.xy_data[goalkeeper][:,0]) > 0:
            team_right = self.team_dict[goalkeeper]
    return team_left, team_right

  def _get_team_field_players(self):
    ## Get the team playing to the left/right using the average value of the player coordinates
    team_left = ''
    team_right = ''
    home_x = self.flip_val*self.input.loc[self.input["team_id"] == self.home_id,'x'].mean(skipna=True)
    away_x = self.flip_val*self.input.loc[self.input["team_id"] == self.away_id,'x'].mean(skipna=True)
    if home_x <  away_x:
      team_left = self.home_away_map[self.home_id]
      team_right = self.home_away_map[self.away_id]
    else:
      team_right = self.home_away_map[self.home_id]
      team_left = self.home_away_map[self.away_id]
    return team_left, team_right


  def _init(self):
    # Initialize all annotations
    for point in self.point_list:
        point[0].set_data([], [])

    self.point_ball[0].set_data([],[])
    self.point_ball_add[0].set_data([],[])

    self.info_time.set_text('')
    self.info_event.set_text('')
    self.info_score.set_text('')
    if self.display_team_stat_text is not None:
        self.info_team_stat.set_text('')
        self.info_team_stat.set_color('black')
    
    return self.point_ball
    

  def _animate(self,i):
    ## Animation function. Called sequentially by the FuncAnimation object
    
    #### EVENT TIME
    self.info_time.set_text(self.display_time[i])
    if self.plot_all_events:
      self.info_score.set_text(self.display_score[i])
      self.info_event.set_text(self.event_annotation[i])
    if self.display_team_stat_text is not None:
        self.info_team_stat.set_text(self.display_team_stat_text[i])
        self.info_team_stat.set_color(self.display_team_stat_color[i])

    if self.ball_status[i] == 'dead':
      self.info_time.set_color('darkgrey')
      self.info_score.set_color('darkgrey')
      self.info_event.set_color('darkgrey')
      if self.display_team_stat_text is not None:
          self.info_team_stat.set_color('darkgrey')
    else:
      self.info_time.set_color('black')
      self.info_score.set_color('black')
      self.info_event.set_color('black')

    
          
    #### OVERLAY EVENTS
    if self.plot_all_events:
      event_color_i = self.event_color[i]
      control_color_i = self.control_color[i]
    
      if self.plot_this_event[i]:
        if self.contest_name[i] == 'contest' or 'duel' in self.contest_name[i]:
          self.point_duel[0].set_data(self.xy_ball[i,0] + self.params.get('pz_duel',2)*np.cos(np.linspace(0,2*np.pi)),
                                      self.xy_ball[i,1] + self.params.get('pz_duel',2)*np.sin(np.linspace(0,2*np.pi)))
          self.point_duel[0].set_color(self.color_events[self.contest_name[i]])
        else:
          self.point_duel[0].set_data([],[])

            
        if self.event_name[i] == 'pass' or self.event_name[i] == 'cross':
          self.pass_counter = 1
          self.passing_player = self.event_player_id[i]
          self.pass_color = event_color_i
        else:
          self.pass_counter += 1
          if self.pass_fade <= self.pass_counter:
            self.point_pass[0].set_data([],[])
            self.passing_player = ''
            
        if 'shot' in self.event_name[i] or 'own' in self.event_name[i]:
          self.shot_counter = 1
          self.shooting_player = self.event_player_id[i]
          self.shot_color = event_color_i 
        else:
          self.shot_counter += 1
          if self.shot_fade <= self.shot_counter:
            self.point_shot[0].set_data([],[])
            self.shooting_player = ''
            self.shot_color = ''

        if any(substring in self.event_name[i] for substring in ['save','claim','loose']):
          self.save_counter = 1
          self.saving_player = self.event_player_id[i]
          self.save_color = event_color_i
        else:
          self.save_counter += 1
          if self.save_fade <= self.save_counter:
            self.point_save[0].set_data([],[])
            self.saving_player = ''
            self.save_color = ''
      else:        
        self.point_duel[0].set_data([],[])
        self.shot_counter += 1
        self.pass_counter += 1
        self.save_counter += 1

        if self.save_fade <= self.save_counter:
          self.point_save[0].set_data([],[])
          self.saving_player = ''

        if self.shot_fade <= self.shot_counter:
          self.point_shot[0].set_data([],[])
          self.shooting_player = ''
          self.shot_color = ''

        if self.pass_fade <= self.pass_counter:
          self.point_pass[0].set_data([],[])
          self.passing_player = ''
    else:
      event_color_i = ''
      control_color_i = dict()

    #### PLOT PLAYERS
    ## Use data from frame j to i for speed effect
    j = max(0, i-self.n)
    j_player = max(0, i-self.n_player)

    for (this_point, this_cz, direction_arrow, this_xy_data, this_visible, this_xy_direction, this_color, this_annot, this_player) in self.player_info:
      if np.isnan(this_xy_data[i,:]).any():
        this_point.set_data([], [])
        this_annot.set_text('')
        if direction_arrow is not None:
            direction_arrow.set_data([],[])
        continue

      if self.plot_pz:
        this_cz.set_center((this_xy_data[i,0], this_xy_data[i,1]))
        this_cz.set_visible(True)

      if self.plot_direction:
        direction_arrow.set_data([this_xy_data[i,0],this_xy_data[i,0]+self.direction_length*this_xy_direction[i][0]],
                                 [this_xy_data[i,1],this_xy_data[i,1]+self.direction_length*this_xy_direction[i][1]])
        if this_visible[i]:
          direction_arrow.set_color(this_color)
        else:
          direction_arrow.set_color(self.color_map['invisible'])


      if np.isnan(this_xy_data[max(0,i-1),:]).any() or i == 0:
        disp_player = 0
      else:
        disp_player = np.sqrt(sum((this_xy_data[i,:] - this_xy_data[i-1,:])**2))
      
      this_point.set_data(this_xy_data[j_player:i+1,0], this_xy_data[j_player:i+1,1])
      if this_visible[i]:
        this_point.set_color(control_color_i.get(this_player,this_color))
      else:
        this_point.set_color(self.color_map['invisible'])

      this_point.set_linestyle('-')
      this_point.set_linewidth(self.player_size)

      if disp_player > 0.05:                      
        this_point.set_marker('')
      else:
        this_point.set_marker('s')
        this_point.set_markersize(self.player_size)
        if this_visible[i]:
          this_point.set_markerfacecolor(control_color_i.get(this_player,this_color))
          this_point.set_markeredgecolor(control_color_i.get(this_player,this_color))
        else:
          this_point.set_markerfacecolor(self.color_map['invisible'])
          this_point.set_markeredgecolor(self.color_map['invisible'])
        
      if self.annotate_tag:
          this_annot.set_color(this_color)

          this_annot.set_position((this_xy_data[i,0]+0.5,this_xy_data[i,1]+0.5))
          this_annot.set_text(self.players_tag.get(this_player,''))

      if self.plot_circles:
          if (this_player == self.passing_player) and (self.pass_fade >= self.pass_counter):
            self.point_pass[0].set_data(this_xy_data[i,0] + 2*np.cos(np.linspace(0,2*np.pi)), this_xy_data[i,1] + 2*np.sin(np.linspace(0,2*np.pi)))
            # print(self.pass_color, self.display_time[i], self.frames[i])
            self.point_pass[0].set_color(self.pass_color)

          if (this_player == self.saving_player) and (self.save_fade >= self.save_counter):
            self.point_save[0].set_data(this_xy_data[i,0] + 2*np.cos(np.linspace(0,2*np.pi)), this_xy_data[i,1] + 2*np.sin(np.linspace(0,2*np.pi)))
            self.point_save[0].set_color(self.save_color)

          if (this_player == self.shooting_player) and (self.shot_fade >= self.shot_counter):
            self.point_shot[0].set_data(this_xy_data[i,0] + 2*np.cos(np.linspace(0,2*np.pi)), this_xy_data[i,1] + 2*np.sin(np.linspace(0,2*np.pi)))
            self.point_shot[0].set_color(self.shot_color)

      
    #### PLOT BALL
    if np.isnan(self.xy_ball[max(0,i-1),:]).any() or i == 0:
      disp_ball = 0 
    else:
      disp_ball = np.sqrt(sum((self.xy_ball[i,:] - self.xy_ball[i-1,:])**2))

    self.point_ball[0].set_data(self.xy_ball[j:i+1,0], self.xy_ball[j:i+1,1])
    if self.ball_visible[i]:
        self.point_ball[0].set_color(self.color_pitch['ball'])
    else:
        self.point_ball[0].set_color(self.color_map['invisible'])
        
    self.point_ball[0].set_linestyle('-')
    self.point_ball[0].set_linewidth(self.ball_size)
    if disp_ball > 0.05:     
      self.point_ball[0].set_marker('')
    else:
      self.point_ball[0].set_marker('o')
      self.point_ball[0].set_markersize(self.ball_size)
      if self.ball_visible[i]:
          self.point_ball[0].set_markerfacecolor(self.color_pitch['ball'])
          self.point_ball[0].set_markeredgecolor(self.color_pitch['ball'])
      else:
          self.point_ball[0].set_markerfacecolor(self.color_map['invisible'])
          self.point_ball[0].set_markeredgecolor(self.color_map['invisible'])
        
    ## Additional ball (if any)
    if self.xy_ball_add is not None:
      if np.isnan(self.xy_ball_add[max(0,i-1),:]).any() or i == 0:
        disp_ball = 0
      else:
        disp_ball = np.sqrt(sum((self.xy_ball_add[i,:] - self.xy_ball_add[i-1,:])**2))

      self.point_ball_add[0].set_data(self.xy_ball_add[j:i+1,0], self.xy_ball_add[j:i+1,1])
      self.point_ball_add[0].set_color(self.color_pitch['ball_add'])
      self.point_ball_add[0].set_linestyle('-')
      self.point_ball_add[0].set_linewidth(self.ball_size)
      if disp_ball > 0.05:     
        self.point_ball_add[0].set_marker('')
      else:
        self.point_ball_add[0].set_marker('o')
        self.point_ball_add[0].set_markersize(self.ball_size)
        self.point_ball_add[0].set_markerfacecolor(self.color_pitch['ball_add'])
        self.point_ball_add[0].set_markeredgecolor(self.color_pitch['ball_add'])

    ## Visualize progress of routine
    if self.verbose:
        if i == 1:
          print("0%")
        else:
          disp_perc = int(np.round(i / len(self.display_time) * 100,0))
          if np.remainder(disp_perc,self.verbose) == 0:
            if disp_perc not in self.progress:
              print(f"{disp_perc}% completed")
              self.progress.append(disp_perc)

    return self.point_ball


