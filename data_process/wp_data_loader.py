import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

class Dataset_wp_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv', 
                target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,self.data_path))

        # 3:1:2 train valid test 
        border1s = [0, 15*30*24 - self.seq_len, 15*30*24+5*30*24 - self.seq_len]
        border2s = [15*30*24, 15*30*24+5*30*24, 15*30*24+5*30*24+10*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data_flag = df_raw[df_raw.columns[0]]
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[2:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        # True
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            train_data_flag = df_data_flag[border1s[0]:border2s[0]]

            train_data_vflag = np.where(train_data_flag.values == 1)
            train_data_v = train_data.values[train_data_vflag]
            train_data_v = train_data_v.astype('float64')
            self.scaler.fit(train_data_v)
            # data = self.scaler.transform(df_data.values)
            # data = self.scaler.fit_transform(df_data.values)
        
        data = df_data.values

        split_stamp = df_raw[['date']][border1:border2]
        split_flag = df_data_flag[border1:border2]
        split_data = data[border1:border2]

        split_stamp['date'] = pd.to_datetime(split_stamp.date)
        data_stamp = time_features(split_stamp, timeenc=self.timeenc, freq=self.freq)
        
        split_vflag = np.where(split_flag.values == 1)
        split_vdata = split_data[split_vflag]
        split_vdata = split_vdata.astype('float64')
        split_vsdata = self.scaler.transform(split_vdata)
        split_vtamp = data_stamp[split_vflag]

        left_borders = []
        for i in range(split_vflag[0].shape[0] - self.seq_len - self.pred_len +1):
            a_begin = i
            a_end = i + self.seq_len + self.pred_len - 1
            if (split_vflag[0][a_end] - split_vflag[0][a_begin]) == (self.seq_len + self.pred_len - 1):
                left_borders.append(a_begin)
            else:
                continue
        
        self.data_x = split_vsdata
        if self.inverse:
            self.data_y = split_vdata
        else:
            self.data_y = split_vsdata
        self.data_stamp = split_vtamp 
        self.left_borders = left_borders

    # index = __len__
    def __getitem__(self, index):
        a_begin = self.left_borders[index]
        s_begin = a_begin
        s_end = a_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]  
        seq_y = self.data_y[r_begin:r_end] 
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.left_borders)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_wp_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            print('error!')

        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,self.data_path))

        border1s = [0, 26000 - self.seq_len, 26000+11000 - self.seq_len]
        border2s = [26000, 26000+11000, 52710]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        df_data_flag = df_raw[df_raw.columns[0]]
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[2:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        # True
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            train_data_flag = df_data_flag[border1s[0]:border2s[0]]

            train_data_vflag = np.where(train_data_flag.values == 1)
            train_data_v = train_data.values[train_data_vflag]
            train_data_v = train_data_v.astype('float64')
            self.scaler.fit(train_data_v)
            # data = self.scaler.transform(df_data.values)
            # data = self.scaler.fit_transform(df_data.values)

        data = df_data.values

        split_stamp = df_raw[['date']][border1:border2]
        split_flag = df_data_flag[border1:border2]
        split_data = data[border1:border2]

        split_stamp['date'] = pd.to_datetime(split_stamp.date)
        data_stamp = time_features(split_stamp, timeenc=self.timeenc, freq=self.freq)

        split_vflag = np.where(split_flag.values == 1)
        split_vdata = split_data[split_vflag]
        split_vdata = split_vdata.astype('float64')
        split_vsdata = self.scaler.transform(split_vdata)
        split_vtamp = data_stamp[split_vflag]

        left_borders = []
        for i in range(split_vflag[0].shape[0] - self.seq_len - self.pred_len +1):
            a_begin = i
            a_end = i + self.seq_len + self.pred_len - 1
            if (split_vflag[0][a_end] - split_vflag[0][a_begin]) == (self.seq_len + self.pred_len - 1):
                left_borders.append(a_begin)
            else:
                continue
        
        self.data_x = split_vsdata
        if self.inverse:
            self.data_y = split_vdata
        else:
            self.data_y = split_vsdata
        self.data_stamp = split_vtamp 
        self.left_borders = left_borders
    
    def __getitem__(self, index):
        a_begin = self.left_borders[index]
        s_begin = a_begin
        s_end = a_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]  
        seq_y = self.data_y[r_begin:r_end] 
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.left_borders) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)