import os
import time
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy import signal

train_set = pd.read_csv('train.csv')
seq_len = 150000
train_len = int(train_set.shape[0] / seq_len * 0.8) * seq_len


def signal_filter(data, mode='lowpass'):
    x = data - data.mean()
    b, a = signal.butter(8, 0.1, mode)
    filted_data = signal.filtfilt(b, a, x)
    return filted_data


def basic_split():
    # basic split
    train_data = train_set.iloc[:train_len, :]
    train_x = [train_data.iloc[i*seq_len:(i+1)*seq_len, 0].values for i in range(int(train_data.shape[0] / seq_len))]
    train_y = [train_data.iloc[(i+1)*seq_len - 1, 1] for i in range(int(train_data.shape[0] / seq_len))]
    train_x = np.stack(train_x)
    train_y = np.array(train_y)
    np.save('train/basic_split/train_x', train_x)
    np.save('train/basic_split/train_y', train_y)

    val_data = train_set.iloc[train_len:, :]
    val_x = [val_data.iloc[i*seq_len:(i+1)*seq_len, 0].values for i in range(int(val_data.shape[0] / seq_len))]
    val_y = [val_data.iloc[(i+1)*seq_len - 1, 1] for i in range(int(val_data.shape[0] / seq_len))]
    val_x = np.stack(val_x)
    val_y = np.array(val_y)
    np.save('train/basic_split/val_x', val_x)
    np.save('train/basic_split/val_y', val_y)


def random_split():
    # random split
    train_data = train_set.iloc[:train_len, :]
    np.save('train/random_split/train_data', train_data.values)

    val_data = train_set.iloc[train_len:, :]
    val_x = [val_data.iloc[i*seq_len:(i+1)*seq_len, 0].values for i in range(int(val_data.shape[0] / seq_len))]
    val_y = [val_data.iloc[(i+1)*seq_len - 1, 1] for i in range(int(val_data.shape[0] / seq_len))]
    val_x = np.stack(val_x)
    val_y = np.array(val_y)
    np.save('train/random_split/val_x', val_x)
    np.save('train/random_split/val_y', val_y)


def get_stat_feature(win_x, win_feature):
    # basic moment feature
    win_feature.append([win_x.mean(), win_x.std(), skew(win_x), kurtosis(win_x)])
    # percentiles feature
    percentiles = np.linspace(0, 100, 11)
    win_feature.append(np.percentile(win_x, percentiles))


def get_t_feature(win_x):
    win_feature = []
    get_stat_feature(win_x, win_feature)
    # window stat feature
    win_stat_feature = []
    small_win_len = 100
    for i in range(int(len(win_x) / small_win_len) * 2):
        small_win = win_x[i*int(small_win_len / 2): i*int(small_win_len / 2)+small_win_len]
        win_stat_feature += [small_win.mean(), small_win.std()]
    win_feature.append(win_stat_feature)
    win_feature = np.concatenate(win_feature)
    return win_feature


def get_f_feature(win_x):
    w = np.fft.fft(win_x)
    shifted_w = np.fft.fftshift(w)
    half_norm_w = (abs(shifted_w) / len(shifted_w))[int(len(shifted_w) / 2):]
    # window stat feature
    win_feature = []
    small_win_len = 150
    for i in range(10):
        small_win = half_norm_w[i*small_win_len: (i+1)*small_win_len]
        get_stat_feature(small_win, win_feature)
    win_feature = np.concatenate(win_feature)
    return win_feature


def get_hand_crafted_feature(data_dir='train/random_split/train_data.npy'):
    train_set = np.load(data_dir)
    seq_len = 150000
    win_len = 3000
    win_num = 50
    ratio = 1
    sample_num = int(train_set.shape[0] / seq_len) * ratio
    all_t_feature, all_f_feature, all_y = [], [], []
    for i in range(sample_num):
        t0 = time.time()
        idx = i * int(seq_len / ratio)
        x = train_set[idx:idx + seq_len, 0]
        x = x - x.mean()  # remove DC
        y = train_set[idx + seq_len - 1, 1]
        t_feature, f_feature = [], []
        for k in range(win_num):
            win_x = x[k * win_len:(k + 1) * win_len]
            t_feature.append(get_t_feature(win_x))
            f_feature.append(get_f_feature(win_x))
        t_feature = np.concatenate(t_feature).astype(np.float32)
        f_feature = np.concatenate(f_feature).astype(np.float32)
        all_t_feature.append(t_feature)
        all_f_feature.append(f_feature)
        all_y.append(y)
        print('{} of {} took {:.4f}s'.format(i + 1, sample_num, time.time() - t0))
    all_t_feature = np.array(all_t_feature)
    all_f_feature = np.array(all_f_feature)
    all_y = np.array(all_y)
    # need to be saved


if __name__ == '__main__':
    # basic_split()
    random_split()
    # get_hand_crafted_feature()

