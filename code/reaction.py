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

def get_reaction(week):
    start = time.time()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    player_dict = {}
    print('starting reaction')
    rootdir = '../data'

    df_plays = pd.read_csv('../data/plays.csv')
    df_games = pd.read_csv('../data/games.csv')
    df_players = pd.read_csv('../data/players.csv')
    df_tr = pd.read_csv('../data/targetedReceiver.csv')

    print('getting reaction for week {}'.format(week))
    df_week = pd.read_csv('../data/week{}.csv'.format(week))
    df_game_play = df_week[['gameId','playId']].groupby(['gameId','playId']).size().reset_index()[['gameId', 'playId']]

    for i, row in df_game_play.iterrows():
        gameId = row['gameId']
        playId = row['playId']
        play_meta_df = h.get_slice(df_plays, playId, gameId)
        if play_meta_df['passResult'].iloc[0] == 'C':
            play_df = h.get_slice(df_week, playId, gameId)
            # Get dataframes for defense and offense throughout play
            offense = play_df[play_df['position'] == 'QB']['team']
            if 'away' in offense.values:
                defense = 'home'
            else:
                defense = 'away'
            # dataframe for defensive players
            def_play_df = play_df[play_df['team'] == defense]
            def_players = def_play_df['nflId'].unique()
            # dataframe for targeted receiver
            target_rec_meta = h.get_slice(df_tr, playId, gameId)
            receiverId = target_rec_meta['targetNflId']
            target_defender = target_rec_meta['defenderId'].squeeze()
            if target_defender > 0:
                receiver_df = play_df[play_df['nflId'] == float(receiverId)]
                target_receiver = receiver_df['nflId'].unique().squeeze()
                # Get frame ids for ball_snap, pass_forward, pass_arrived
                try:
                    frame_ball_snap = play_df[play_df['event'] == 'ball_snap']['frameId'].iloc[0]
                    frame_pass_fwd = play_df[play_df['event'] == 'pass_forward']['frameId'].iloc[0]
                    frame_pass_arrived = play_df[play_df['event'] == 'pass_arrived']['frameId'].iloc[0]
                except:
                    continue
                target_rec_df = receiver_df[receiver_df['frameId'].isin(list(range(frame_pass_fwd,len(receiver_df)+1)))]
                # Loop through defense players to estimate the time to switch to target receiver pursuit
                for playerId in def_players:
                    times = []
                    if playerId != target_defender:
                        opt_route_def_df = def_play_df[def_play_df['nflId'] == playerId]
                        if playerId not in player_dict:
                            player_dict[playerId] = {'name':opt_route_def_df['displayName'].iloc[0], 'reaction_rate':[], 'time_to_react':[]}
                        # Get T_opt for moment of pass forward
                            end_frame = min(receiver_df['frameId'].max(), opt_route_def_df['frameId'].max())+1
                        try:
                            opt_route_def_df = opt_route_def_df[opt_route_def_df['frameId'].isin(list(range(frame_pass_fwd,end_frame)))]
                        except:
                            ipdb.set_trace()
                        if len(opt_route_def_df) >0:
                            frame, time_elapsed, intersection = ot.get_frame_intersection(opt_route_def_df, target_rec_df)
                        else:
                            frame = False
                        if frame != False:
                            times.append(time_elapsed)
                            # If possible for defender to reach receiver within play's end, check for T'_opt
                            if frame+1 < end_frame:
                                # step along defender's and offender's trajectories to final new time_elapsed at moment T_t
                                ct = 0
                                for i in range(frame_pass_fwd+1, end_frame):
                                    if frame_pass_fwd+1+ct < end_frame:
                                        new_route_def_df = opt_route_def_df[opt_route_def_df['frameId'].isin(list(range(frame_pass_fwd+1+ct,end_frame)))]
                                        new_target_rec_df = receiver_df[receiver_df['frameId'].isin(list(range(frame_pass_fwd+1+ct,end_frame)))]
                                        # update frame to be T'_opt intersection frame
                                        frame, time_elapsed, intersection = ot.get_frame_intersection(new_route_def_df, new_target_rec_df)
                                        # Append to times for plot
                                        times.append(time_elapsed)
                                        ct += 1
                                    else:
                                        break
                                # Create x's for each frame
                                x = np.array(range(len(times)))/10
                                # remove points where Topt = max time
                                times = times + x
                                # plt.plot(x,times)
                                max_time = max(times)
                                mask = np.where(times == max_time)
                                times = np.delete(times, mask)
                                x = np.delete(x, mask)
                                # calculate average slope along T'_opt/dt line
                                x_s = np.array(x)
                                y_s = np.array(times)
                                dy = y_s[:-1] - y_s[1:]
                                dx = x_s[:-1] - x_s[1:]
                                slope = dy/dx
                                end_of_play = max(times)
                                flats = np.where(slope < .01)
                                not_top = np.where(times < end_of_play)
                                react_time = flats[0][np.argmax(np.isin(flats, not_top))] *.1
                                ave_slope = np.mean(slope)
                                player_dict[playerId]['reaction_rate'].append(float(ave_slope))
                                player_dict[playerId]['time_to_react'].append(float(react_time))
            # title = play_meta_df['playDescription'].squeeze()
            # plt.title(title)
            # plt.savefig('metrics/results/plots/{}-{}.png'.format(gameId, playId))
            # plt.close()
    output_df = pd.DataFrame(player_dict).T
    output_df.to_csv('../results/week{}reaction.csv'.format(week))
    end = time.time()
    print('Time elapsed {}'.format(end-start))

def compile_reaction():
    rootdir = '../results/reaction'
    player_dict = {}
    final_player_dict = {}
    for i in range(1,18):
        df_week_hrr = util.load_csv(rootdir, 'week{}reaction.csv'.format(i))
        for idx, player in df_week_hrr.iterrows():
            hrr = player['reaction_rate']
            if len(hrr)>2:
                name = player['name']
                hrr = hrr[1:-1]
                hrr = [float(x.strip()) for x in hrr.split(',')]
                nflId = player[0]
                if nflId not in player_dict.keys():
                    player_dict[nflId] = {'name': name, 'hrr': []}
                player_dict[nflId]['hrr'].extend(hrr)
    for idx, player in player_dict.items():
        name = player['name']
        hrr = player['hrr']
        total_hrr = sum(hrr)
        hrr_per_game = total_hrr/len(hrr)
        med_hrr = np.median(hrr)
        final_player_dict[idx]=({'nflId': idx, 'name':name, 'med_hrr': med_hrr, 'average_hrr': hrr_per_game})
    output_df = pd.DataFrame(final_player_dict).T
    output_df.to_csv('../results/reaction/hrr.csv', index=False)

def main():
    Parallel(n_jobs = 2)(delayed(get_reaction)(i) for i in range(1,18))

if __name__ == "__main__:":
    main()