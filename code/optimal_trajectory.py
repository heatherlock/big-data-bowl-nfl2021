import keras
import sys
import linecache
import util
import ipdb

import pandas as pd
import numpy as np
import tensorflow as tf

def get_frame_intersection(df_defender, df_offender):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    model = keras.models.load_model('time_model.h5')
    # model.summary()
    # Get coords of defender at first frame
    try:
        defenderX = df_defender['x'].iloc[0]
    except:
        return (False, False, False)
    defenderY = df_defender['y'].iloc[0]
    defender_dir = df_defender['dir'].iloc[0]
    # Get distance from defender along offender's trajectory
    distances = np.array(np.sqrt((df_offender['x'] - defenderX)**2 + (df_offender['y'] - defenderY)**2))
    # distances = distances.reshape(1,len(distances))
    speeds = np.array([df_defender['s'].iloc[0].squeeze()]*len(df_defender))
    # speeds = speeds.reshape(1,len(speeds))
    # Compute angle between defender's original direction and possible point of intersection
    def_x = np.array([defenderX]*len(df_defender))
    def_y = np.array([defenderY]*len(df_defender))
    off_x = np.array(df_offender['x'])
    off_y = np.array(df_offender['y'])
    defender_vector = np.array([defender_dir] * len(df_defender))
    angles = get_angle(defender_vector, def_x, def_y, off_x, off_y) 
    if type(angles) is np.ndarray:
        try: 
            inputs = np.vstack((distances, speeds, angles)).T
        except:
            return (False, False, False)
        # get predictions along route
        y_pred = model.predict(inputs)
        # get time elapsed since pass_forward frame
        times = pd.to_datetime(df_offender['time'], format='%Y-%m-%dT%H:%M:%S.%fZ') 
        time_elapsed = (times[1:] - times.iloc[0])
        time_elapsed = np.array(time_elapsed.dt.total_seconds())
        time_elapsed = np.insert(time_elapsed, 0, 0.0)
        max_frame = df_offender['frameId'].max() - df_offender['frameId'].min()
        intersection_idx = [max_frame, max_frame]
        for i in range(len(time_elapsed)-1,0,-1):
            for j, time in enumerate(y_pred[:,0]):
                if time - time_elapsed[i] <= 0.0:
                    intersection_idx = [i,j]
                    break
        frame = df_defender['frameId'].iloc[intersection_idx[0]]
        time = time_elapsed[intersection_idx[0]]
        off_xy = df_offender[df_offender['frameId'] == frame][['x','y']]
    else:
        # something got messed up in angles
        return (False, False, False)
    return (frame, time, off_xy)

def get_angle(init_dir, defX, defY, offX, offY):
    try:
        init_angle = np.radians(init_dir)
        d_x = offX - defX
        d_y = offY - defY
        theta = np.arctan2(np.cos(init_angle) * d_y - d_x * np.sin(init_angle), d_x * np.cos(init_angle) + d_y * np.sin(init_angle))
    except:
        return False
    return theta

def main():
    # optimal trajectory for Jacob Crowder touchdown
    rootdir = '../data/'
    gameId = 2018120908
    playId = 3577
    df_week = util.load_csv(rootdir, 'week14.csv')
    df_plays = util.load_csv(rootdir, 'plays.csv')
    df_games = util.load_csv(rootdir, 'games.csv')
    df_target = util.load_csv(rootdir, 'target_defender.csv')

    target = util.get_slice_by_id(df_target, playId, gameId)
    targetId = target['targetNflId'].squeeze()

    df_tracks = util.get_slice_by_id(df_week, playId, gameId)
    df_receiver = df_tracks[df_tracks['nflId'] == targetId]
    df_defender = df_tracks[df_tracks['nflId'] == 2561318.0]
    frame_pf = df_receiver[df_receiver['event'] == 'pass_forward']['frameId'].squeeze()
    df_defender_pf = df_defender[df_defender['frameId'] == frame_pf]
    print([df_defender_pf['x'].squeeze(),df_defender_pf['y'].squeeze()])
    # Make frames from pass_forward to end
    end_frame = min(len(df_defender), len(df_receiver))+1
    df_defender = df_defender[df_defender['frameId'].isin(list(range(frame_pf, end_frame)))]
    df_receiver = df_receiver[df_receiver['frameId'].isin(list(range(frame_pf, end_frame)))]

    frame, time, off_xy = get_frame_intersection(df_defender, df_receiver)
    print(frame)
    print(time)
    print(off_xy)

if __name__ == "__main__":
    main()