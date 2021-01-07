import ipdb
import matplotlib
import time
import json

import multiprocessing as mp
import tensorflow as tf
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import optimal_trajectory as ot
import matplotlib.pyplot as plt

import helper as h

"""This module computes yards allowed after optimal intersection (YATI) per player for each week."""

def get_yati(week):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    player_dict = {}
    print('starting yac')
    rootdir = '../data/'

    df_plays = pd.read_csv('../data/plays.csv')
    df_games = pd.read_csv('../data/games.csv')
    df_players = pd.read_csv('../data/players.csv')
    df_tr = pd.read_csv('../data/target_deceiver.csv')

    print('getting yac for week {}'.format(week))
    df_week = util.load_csv(rootdir, 'week{}.csv'.format(week))
    df_game_play = df_week[['gameId','playId']].groupby(['gameId','playId']).size().reset_index()[['gameId', 'playId']]

    for i, row in df_game_play.iterrows():
        gameId = row['gameId']
        playId = row['playId']
        play_meta_df = util.get_slice_by_id(df_plays, playId, gameId)
        if play_meta_df['passResult'].iloc[0] == 'C':
            play_df = util.get_slice_by_id(df_week, playId, gameId)

            receiverId = util.get_slice_by_id(df_tr, playId, gameId)['targetNflId'].squeeze()
            receiver_df = play_df[play_df['nflId'] == float(receiverId)]

            # Get dataframes for defense and offense throughout play
            # Do not use if no targeted receiver
            try:
                offense = receiver_df['team'].unique()[0]
            except:
                continue
            if offense == 'away':
                defense = 'home'
            else:
                defense = 'away'

            # dataframe for defensive players
            def_play_df = play_df[play_df['team'] == defense]
            def_players = def_play_df['nflId'].unique()
            # dataframe for targeted receiver
            target_rec_meta = util.get_slice_by_id(df_tr, playId, gameId)
            receiverId = target_rec_meta['targetNflId']
            target_defender = target_rec_meta['defenderId'].squeeze()
            receiver_df = play_df[play_df['nflId'] == float(receiverId)]
            target_receiver = receiver_df['nflId'].unique().squeeze()
            
            # Get frame ids for ball_snap, pass_forward, pass_arrived
            # do not use if event missing
            try:
                frame_ball_snap = play_df[play_df['event'] == 'ball_snap']['frameId'].iloc[0]
                frame_pass_fwd = play_df[play_df['event'] == 'pass_forward']['frameId'].iloc[0]
                frame_pass_arrived = play_df[play_df['event'] == 'pass_arrived']['frameId'].iloc[0]
            except:
                continue

            lines = []
            intersections = []
            pass_forward_xy = []
            intersection_frames = {}
            target_rec_df = receiver_df[receiver_df['frameId'].isin(list(range(frame_pass_fwd,len(receiver_df))))]
            # Loop through defense players to estimate their ideal trajectory to target receiver from moment of pass forward
            # and yards allowed
            for playerId in def_players:
                if playerId != target_defender:
                    opt_route_def_df = def_play_df[def_play_df['nflId'] == playerId]
                    if playerId not in player_dict:
                        try:
                            player_dict[playerId] = {'name':opt_route_def_df['displayName'].iloc[0], 'YACNTR':0.0, 'Missed Opps': 0.0, 'weeks':[]}
                        except:
                            continue
                    # Get optimality of route
                    opt_route_def_df = opt_route_def_df[opt_route_def_df['frameId'].isin(list(range(frame_pass_fwd,len(receiver_df))))]
                    if len(opt_route_def_df) >0:
                        frame, time_elapsed, intersection = ot.get_frame_intersection(opt_route_def_df, target_rec_df)
                    else:
                        frame = False
                    if frame != False:
                        intersections.append(intersection.squeeze().to_numpy())
                        intersection_frames[playerId] = int(frame)
                        pass_forward_xy.append(opt_route_def_df[opt_route_def_df['frameId'] == frame_pass_fwd][['x','y','displayName']].squeeze().to_numpy())
                        # Return yards allowed
                        pass_arrived_x = target_rec_df[target_rec_df['event'] == 'pass_arrived']['x'].squeeze()
                        # TODO: Adjust for just crossing endzone line
                        final_x = target_rec_df['x'].iloc[len(target_rec_df)-1]
                        # If intersection point occurs before pass_arrived, then yards count from pass_arrived to end, otherwise from point of intersection.
                        # If point occurs before pass_arrived, theoretically, they could have forced an incomplete so possible missed possible forced incomplete +1
                        if frame < frame_pass_arrived:
                            yards_allowed = abs(final_x - pass_arrived_x)
                            player_dict[playerId]['YACNTR'] += yards_allowed
                            player_dict[playerId]['Missed Opps'] += 1
                        else:
                            yards_allowed = abs(final_x - intersection['x'].squeeze())
                            player_dict[playerId]['YACNTR'] += yards_allowed
                        if week not in player_dict[playerId]['weeks']:
                            player_dict[playerId]['weeks'].append(week)
            play_desc = play_meta_df['playDescription'].squeeze()
            # Only use if not in parallel!!!
            # display_intersection(intersection_frames, receiver_df, def_play_df, play_desc)
            ipdb.set_trace()
    output_df = pd.DataFrame(player_dict).T
    output_df.to_csv('../results/week{}yac.csv'.format(week))

def compile_yati():
    """Compiles weekly YATI csvs into 1 that computes YATI and MO per game"""
    rootdir = '../results/yati'
    player_dict = {}
    final_player_dict = {}
    for i in range(1,18):
        df_week_yac = util.load_csv(rootdir, 'week{}yac.csv'.format(i))
        for idx, player in df_week_yac.iterrows():
            name = player['name']
            yaoi = player['YACNTR']
            nflId = player['nflId']
            mo = player['Missed Opps']
            if nflId not in player_dict.keys():
                player_dict[nflId] = {'name': name, 'yaoi': [], 'mo': []}
            player_dict[nflId]['yaoi'].append(yaoi)
            player_dict[nflId]['mo'].append(mo)
    for idx, player in player_dict.items():
        nflId = idx
        name = player['name']
        yaoi = player['yaoi']
        mo = player['mo']
        total_yaoi = sum(yaoi)
        yaoi_per_game = total_yaoi/len(yaoi)
        mo_per_game = sum(mo)/len(mo)
        total_mo = sum(mo)
        final_player_dict[nflId]=({'id': nflId, 'name':name, 'totalYATI': total_yaoi, 'YATIperGame': yaoi_per_game, 'MOperGame':mo_per_game, 'totalMO':total_mo})
    output_df = pd.DataFrame(final_player_dict).T
    output_df.to_csv('../results/yati/yati.csv')

def display_intersection(intersection_frames, df_tr, def_play_df, desc):
    # Displays the play and moments of intersection from defender to receiver
    # Plot Field bounds
    plt.scatter([0,120],[0,53.3])

    # Plot Receiver's trajectory
    rec_x = list(df_tr['x'])
    rec_y = list(df_tr['y'])
    plt.plot(rec_x, rec_y, color='orange')

    events = df_tr['event'].unique()
    for event in events:
        if event != 'None':
            event_row = df_tr[df_tr['event'] == event]
            x = event_row['x']
            y = event_row['y']
            plt.scatter(x,y,marker='o', color='orange')
            plt.annotate(event, (x, y))
        # Plot defender's trajectories
    defenders = intersection_frames.keys()
    for defender in defenders:
        # Whole trajectory
        plot_df = def_play_df[def_play_df['nflId'] == defender]
        name = plot_df['displayName'].iloc[0]
        plt.plot(plot_df['x'], plot_df['y'])
        # Intersection lines for earliest arrival point
        frame_pass_fwd = def_play_df[def_play_df['event'] == 'pass_forward']['frameId'].iloc[0]
        pf_df = plot_df[plot_df['frameId'] == frame_pass_fwd]
        int_df = df_tr[df_tr['frameId'] == intersection_frames[defender]]
        # x,y of intersection
        int_x = int_df['x'].squeeze()
        int_y = int_df['y'].squeeze()
        # x,y coordinate of defender at moment of pass forward
        pf_x = plot_df[plot_df['frameId'] == intersection_frames[defender]]['x']
        pf_y = plot_df[plot_df['frameId'] == intersection_frames[defender]]['y'] 
        plt.plot([pf_x, int_x], [pf_y, int_y], marker='x', linestyle='dashed')
        plt.annotate(name, (plot_df['x'].iloc[0], plot_df['y'].iloc[0])) 
    plt.title(desc)   
    plt.show()

def main():
    Parallel(n_jobs = 2)(delayed(get_yati)(i) for i in range(1,18))

if __name__ == "__main__":
    main()