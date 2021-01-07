import ipdb
import matplotlib
import datetime

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


def slice_frame(df, playId, gameId):
    tmp = df[df['gameId'] == gameId]
    return tmp[tmp['playId'] == playId]

def get_closest_opposition(defender_df, offense_df):
    try:
        # offense_df is frame of oppposing team, defender_df is just one defender's frame
        distance_dict = {}
        distance_dict_weighted = {}
        offenders = offense_df['nflId'].unique()
        # Loop through offensive players to multiply dis * weights per frame, and get average distance throughout input trajectory
        for offenderId in offenders:
            if not np.isnan(offenderId):
                offender_df = offense_df[offense_df['nflId'] == offenderId]
                offender_df['dis'] = np.sqrt((offender_df['x'].to_numpy() - defender_df['x'].to_numpy())**2 + (offender_df['y'].to_numpy() - defender_df['y'].to_numpy())**2)
                num_frames = len(offender_df)
                weights = np.array(np.arange(2.0,(num_frames+1)*2.0,2.0)) # weights per frame, weights are greater near end of trajectory
                # for i in range(len(offender_df)):
                average = offender_df['dis'].to_numpy().sum()/len(offender_df['dis'])
                weighted_distance = offender_df['dis'].to_numpy() * weights
                weighted_average = weighted_distance.sum()/weights.sum()
                distance_dict[offenderId] = average
                distance_dict_weighted[offenderId] = weighted_average
        # Assign min as chosen opponent
        opponentId = min(distance_dict_weighted, key=distance_dict_weighted.get)
        # Get actual average distance for chosen opponent
        average_distance = distance_dict[opponentId]
    except:
        opponentId = 0
        average_distance = 1000
    return opponentId, average_distance

def get_time(date_time_string):
    date_time_obj = datetime.datetime.strptime(date_time_string, '%Y-%m-%dT%H:%M:%S.%fZ')
    return date_time_obj
