import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import torch
from torch.utils.data import Dataset, DataLoader

np.random.seed(7)
torch.manual_seed(7)


class EarthquakeDataset(Dataset):
    def __init__(self, data_dir, dataset_type, use_all=False,
                 seq_len=150000, win_len=3000, sample_rate=10,
                 use_t=True, use_f=False, win_hand_t=False, win_hand_f=False):
        super(EarthquakeDataset, self).__init__()
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.use_all = use_all
        self.seq_len = seq_len
        self.win_len = win_len
        self.win_num = int(seq_len / win_len)
        self.sample_rate = sample_rate
        self.use_t = use_t
        self.use_f = use_f
        self.win_hand_t = win_hand_t
        self.win_hand_f = win_hand_f

        if use_all:
            self.dataset = np.load(os.path.join(data_dir, 'train_data.npy')).astype(np.float32)
        else:
            if dataset_type == 'test':
                test_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
                ids, values = [], []
                for f in test_files:
                    test_data = pd.read_csv(os.path.join(data_dir, f))
                    ids.append(f[:-4])
                    values.append(test_data.acoustic_data.values.astype(np.float32))
                self.x = np.stack(values)
                self.ids = ids
            else:
                self.x = np.load(os.path.join(data_dir, '{}_x.npy'.format(dataset_type))).astype(np.float32)
                self.y = np.load(os.path.join(data_dir, '{}_y.npy'.format(dataset_type))).astype(np.float32)

    def __getitem__(self, index):
        if self.use_all:
            s_idx = index * self.seq_len
            r_idx = np.random.randint(int(self.seq_len / 2))
            idx = s_idx + r_idx
            x = self.dataset[idx:idx+self.seq_len, 0]
            y = self.dataset[idx+self.seq_len-1, 1]
        else:
            if self.dataset_type == 'test':
                x = self.x[index]
                y = self.ids[index]
            else:
                x = self.x[index]
                y = self.y[index]

        x = x - x.mean()  # remove DC
        t_feature, f_feature = [], []
        if self.use_t:
            if not self.win_hand_t:
                t_feature = x[::self.sample_rate]
            else:
                for i in range(self.win_num):
                    win_x = x[i * self.win_len:(i + 1) * self.win_len]
                    t_feature += get_stat_feature(win_x)
                    # down-sampled original signal
                    t_feature.append(win_x[::50])
                t_feature = np.concatenate(t_feature).astype(np.float32)

        if self.use_f:
            if not self.win_hand_f:
                f_feature = [get_freq_feature(x[i*self.win_len:(i+1)*self.win_len]) for i in range(self.win_num)]
                f_feature = np.concatenate(f_feature).astype(np.float32)
                f_feature = f_feature[::self.sample_rate]
            else:
                for i in range(self.win_num):
                    win_x = x[i*self.win_len:(i+1)*self.win_len]
                    half_norm_w = get_freq_feature(win_x)
                    f_feature += get_stat_feature(half_norm_w)
                    # down-sampled original signal
                    f_feature.append(half_norm_w[::25])
                f_feature = np.concatenate(f_feature).astype(np.float32)

        out = [t_feature, f_feature, y]
        return out

    def __len__(self):
        if self.use_all:
            return int(self.dataset.shape[0] / self.seq_len) - 1
        else:
            return len(self.x)


def get_freq_feature(win_x):
    w = np.fft.fft(win_x)
    shifted_w = np.fft.fftshift(w)
    half_norm_w = (abs(shifted_w) / len(shifted_w))[int(len(shifted_w) / 2):]
    return half_norm_w


def get_stat_feature(win_x):
    win_feature = []
    # basic moment feature
    win_feature.append([win_x.mean(), win_x.std(), skew(win_x), kurtosis(win_x)])
    # percentiles feature
    percentiles = np.linspace(0, 100, 11)
    win_feature.append(np.percentile(win_x, percentiles))
    return win_feature


def dataloader(data_dir, dataset_type, use_all,
               batch_size, shuffle, num_workers=4,
               seq_len=150000, win_len=3000, sample_rate=10,
               use_t=True, use_f=False, win_hand_t=False, win_hand_f=False):

    dset = EarthquakeDataset(data_dir, dataset_type, use_all,
                             seq_len=seq_len, win_len=win_len, sample_rate=sample_rate,
                             use_t=use_t, use_f=use_f, win_hand_t=win_hand_t, win_hand_f=win_hand_f)
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers)
    return dset, loader
