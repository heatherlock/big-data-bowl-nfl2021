import ipdb
import matplotlib
import time
import json

import multiprocessing as mp
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import helper as h

def display_traj(play_df, defenderId, receiverId, offense):
    """Displays trajectory of receiver, defender pair chosen by coverage function
    as well as the rest of the offense and defense in lighter colors"""
    players = play_df['nflId'].dropna().unique()
    for playerId in players:
        player_df = play_df[play_df['nflId'] == playerId]
        team = player_df['team'].iloc[0]
        x = player_df['x']
        y = player_df['y']
        if playerId == defenderId:
            c = 'r'
            l = 'defender'
        elif playerId == receiverId:
            c = 'b'
            l = 'receiver'
        elif team == offense:
            c = 'lightblue'
            l = 'other receivers'
        else:
            c = 'gray'
            l = 'other defenders'
        plt.plot(x,y, color=c, label=l)
        plt.legend()

def get_coverage():
    # Load data
    rootdir = '../data'
    df_plays = pd.read_csv('../data/plays.csv')
    df_games = pd.read_csv('../data/games.csv')
    df_players = pd.read_csv('../data/players.csv')
    df_tr = pd.read_csv('../data/targetedReceiver.csv')
    df_tr = df_tr.dropna().reset_index()
    df_tr['defenderId'] = 0
    for week in range(1,18):
        useable_plays = 0
        total_plays = 0
        player_dict = {}
        print('getting coverage')
        print('starting coverage for week {}'.format(week))
        df_week = pd.read_csv('../data/week{}.csv'.format(week))
        df_game_play = df_week[['gameId','playId']].groupby(['gameId','playId']).size().reset_index()[['gameId', 'playId']]   

        for i, row in df_game_play.iterrows():
            total_plays += 1
            gameId = row['gameId']
            playId = row['playId']
            play_meta_df = h.slice_frame(df_plays, playId, gameId)
            if play_meta_df['passResult'].iloc[0] in ['I', 'C']:
                # print('getting coverage for game: {}, play {}'.format(gameId, playId))
                play_df = h.slice_frame(df_week, playId, gameId)
                play_desc = play_meta_df['playDescription'].squeeze()
                # Get targeted receiver's data frame
                try:
                    # Do not use if no targeted receiver
                    receiverId = h.slice_frame(df_tr, playId, gameId)['targetNflId'].squeeze()
                    receiver_df = play_df[play_df['nflId'] == float(receiverId)]
                    receiver_pos = receiver_df['position'].unique()[0]
                except:
                    continue
                #  Do not use in targeted receiver is not on offense
                if receiver_pos[0] not in ['WR', 'HB', 'FB', 'TE']:
                    # Get frame ids for ball_snap, pass_forward, pass_arrived, do not use if no pass_forward or pass_arrived events
                    frame_ball_snap = play_df[play_df['event'] == 'ball_snap']['frameId'].iloc[0]
                    try:
                        frame_pass_fwd = play_df[play_df['event'] == 'pass_forward']['frameId'].iloc[0]
                        frame_pass_arrived = play_df[play_df['event'] == 'pass_arrived']['frameId'].iloc[0]
                    except:
                        continue
                    # Get dataframes for defense and offense throughout play
                    try:
                        offense = receiver_df['team'].unique()[0]
                    except:
                        continue
                    if offense == 'away':
                        defense = 'home'
                    else:
                        defense = 'away'

                    phase1_df = play_df[play_df['frameId'].isin(list(range(frame_ball_snap,frame_pass_fwd)))]
                    phase1_defense_df = phase1_df[phase1_df['team'] == defense]
                    def_players = phase1_defense_df['nflId'].unique().tolist()
                    phase1_offense_df = phase1_df[phase1_df['team'] == offense]
                    off_players = phase1_offense_df['nflId'].unique()
                    phase1_tr_df = receiver_df[receiver_df['frameId'].isin(list(range(frame_ball_snap,frame_pass_fwd)))]

                    # Get closest defender for targeted receiver
                    opponentId, ave_distance = h.get_closest_opposition(phase1_tr_df, phase1_defense_df)
                    phase1_defender_df = phase1_defense_df[phase1_defense_df['nflId'] == opponentId]
                    # Set defender as target defender in target_defender df
                    game_df_tr = df_tr[df_tr['gameId'] == gameId]
                    game_play_df_tr = game_df_tr[game_df_tr['playId'] == playId]
                    idx = game_play_df_tr.index.values[0]
                    df_tr['defenderId'].iloc[idx] = opponentId  
                    if opponentId not in player_dict:
                        try:
                            player_dict[opponentId] = {'name':phase1_defender_df['displayName'].iloc[0], 'coverage':[]}
                        except:
                            continue
                    player_dict[opponentId]['coverage'].append(ave_distance)
                    # Remove target receiver and their defender for future coverage assignments
                    def_players.remove(opponentId)
                    phase1_offense_df = phase1_offense_df[phase1_offense_df['nflId'] != receiverId]
                    useable_plays += 1
                    # fig = plt.figure(1, figsize=(10,6))
                    for playerId in def_players:
                        # Get df for that player during phase 1
                        phase1_defender_df = phase1_defense_df[phase1_defense_df['nflId'] == playerId]
                        if playerId not in player_dict:
                            player_dict[playerId] = {'name':phase1_defender_df['displayName'].iloc[0],
                            'coverage':[]}
                        # Get nearest opponent on average during phase 1
                        opponentId, ave_distance = h.get_closest_opposition(phase1_defender_df, phase1_offense_df)
                        # display_traj(play_df, playerId, opponentId, offense)
                        # plt.pause(1)
                        # plt.clf()
                        if opponentId != 0:
                            play_defender_df = phase1_defense_df[phase1_defense_df['nflId'] == playerId]                    
                            player_dict[playerId]['coverage'].append(ave_distance)
        print('Week {} useable plays: {}'.format(week, useable_plays))
        output_df = pd.DataFrame(player_dict).T
        output_df['nFlId'] = output_df.index
        output_df.to_csv('../results/coverage/cov_week{}.csv'.format(week), index=False)
    # with open('../results/coverage/week{}_coverage.json'.format(week), 'w') as f:
    #     json.dump(player_dict, f, indent=4)
    df_tr.to_csv('../data/target_defender.csv')
    print('Total useable plays: {}'.format(useable_plays))

def compile_cov():
    """Takes coverage from each week and compiles into one csv and computes median and mean per player"""
    rootdir = '../results/coverage'
    player_dict = {}
    final_player_dict = {}
    for i in range(1,18):
        with open('../results/coverage/week{}_coverage.json'.format(i), 'r') as f:
            data = json.load(f)
        for idx, player in data.items():
            name = player['name']
            cov = player['coverage']
            nflId = idx
            if nflId not in player_dict.keys():
                player_dict[nflId] = {'name': name, 'coverage': []}
            player_dict[nflId]['coverage'].extend(cov)
    for idx, player in player_dict.items():
        nflId = idx
        name = player['name']
        cov = player['coverage']
        total_cov = sum(cov)
        cov_per_game = total_cov/len(cov)
        med_cov = np.median(cov)
        final_player_dict[nflId]=({'nflId': nflId, 'name':name, 'med_coverage': med_cov, 'average_coverage': cov_per_game, 'numPlays':len(cov)})
    output_df = pd.DataFrame(final_player_dict).T
    output_df.to_csv('../results/coverage/coverage.csv', index=False)

def main():
    get_coverage()

if __name__ == "__main__":
    main()