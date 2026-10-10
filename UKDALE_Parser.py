import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from NILM_Dataset import NILMDataset
from Pretrain_Dataset import Pretrain_Dataset


class UK_Dale_Parser:
    
    def __init__(self, args, stats=None):
        self.data_location   = args.ukdale_location
        self.house_indicies  = args.house_indicies
        self.appliance_names = args.appliance_names
        self.sampling        = args.sampling
        self.normalize       = args.normalize
        
        self.cutoff        = [args.cutoff[appl] for appl in ['aggregate'] + args.appliance_names]
        self.threshold     = [args.threshold[appl] for appl in args.appliance_names]
        self.min_on        = [args.min_on[appl] for appl in args.appliance_names]
        self.min_off       = [args.min_off[appl] for appl in args.appliance_names]

        self.val_size      = args.validation_size
        self.window_size   = args.window_size
        self.window_stride = args.window_stride

        self.x, self.y, self.status = self.load_data_with_status()
        
        if self.normalize == 'mean':
            if stats is None:
                self.x_mean = np.mean(self.x)
                self.x_std  = np.std(self.x)
            else:
                self.x_mean, self.x_std = stats
            self.x = (self.x - self.x_mean) / self.x_std
        elif self.normalize == 'minmax':
            if stats is None:
                self.x_min = min(self.x)
                self.x_max = max(self.x)
            else:
                self.x_min, self.x_max = stats
            self.x = (self.x - self.x_min) / (self.x_max - self.x_min)

    def load_data_with_status(self):
        """加载数据并返回 x, y, status（status 已经过 min_on/min_off 过滤）"""
        all_dfs = []
        directory = Path(self.data_location)
        
        for house_id in self.house_indicies:
            house_folder = directory / f'house_{house_id}'
            house_label  = pd.read_csv(house_folder / 'labels.dat', sep=' ', header=None)
            house_data   = pd.read_csv(house_folder / 'channel_1.dat', sep=' ', header=None)
            
            house_data.columns = ['time', 'aggregate']
            house_data['time'] = pd.to_datetime(house_data['time'], unit='s')
            house_data = house_data.set_index('time').resample(self.sampling).mean().fillna(method='ffill', limit=30)
            
            appliance_list = house_label.iloc[:, 1].values
            app_index_dict = defaultdict(list)
            
            for appliance in self.appliance_names:
                try:
                    idx = appliance_list.tolist().index(appliance)
                    app_index_dict[appliance].append(idx + 1)
                except ValueError:
                    app_index_dict[appliance].append(-1)
            
            if np.sum(list(app_index_dict.values())) == -len(self.appliance_names):
                continue
                
            for appliance in self.appliance_names:
                channel_idx = app_index_dict[appliance][0]
                if channel_idx == -1:
                    house_data.insert(len(house_data.columns), appliance, np.zeros(len(house_data)))
                else:
                    channel_path = house_folder / f'channel_{channel_idx}.dat'
                    appl_data = pd.read_csv(channel_path, sep=' ', header=None)
                    appl_data.columns = ['time', appliance]
                    appl_data['time'] = pd.to_datetime(appl_data['time'], unit='s')
                    appl_data = appl_data.set_index('time').resample(self.sampling).mean().fillna(method='ffill', limit=30)
                    house_data = pd.merge(house_data, appl_data, how='inner', on='time')
            
            all_dfs.append(house_data)
        
        if not all_dfs:
            raise ValueError("未加载到任何房屋数据")
        
        entire_data = pd.concat(all_dfs, ignore_index=False)
        entire_data = entire_data.dropna().copy()
        entire_data = entire_data[entire_data['aggregate'] > 0]
        entire_data[entire_data < 5] = 0
        entire_data = entire_data.clip([0] * len(entire_data.columns), self.cutoff, axis=1)
        
        x = entire_data['aggregate'].values.astype(np.float32)
        y = entire_data[self.appliance_names[0]].values.astype(np.float32)
        # 计算状态（使用 min_on/min_off）
        status = self.compute_status(y)
        return x, y, status

    def compute_status(self, data):
        initial_status = data >= self.threshold[0]
        status_diff = np.diff(initial_status)
        events_idx = status_diff.nonzero()
        events_idx = np.array(events_idx).squeeze()
        events_idx += 1

        if initial_status[0]:
            events_idx = np.insert(events_idx, 0, 0)
        if initial_status[-1]:
            events_idx = np.insert(events_idx, events_idx.size, initial_status.size)

        if len(events_idx) < 2:
            return np.zeros_like(data)

        events_idx = events_idx.reshape((-1, 2))
        on_events = events_idx[:, 0].copy()
        off_events = events_idx[:, 1].copy()
        assert len(on_events) == len(off_events)

        if len(on_events) > 0:
            off_duration = on_events[1:] - off_events[:-1]
            off_duration = np.insert(off_duration, 0, 1000)
            on_events = on_events[off_duration > self.min_off[0]]
            off_events = off_events[np.roll(off_duration, -1) > self.min_off[0]]

            on_duration = off_events - on_events
            on_events = on_events[on_duration >= self.min_on[0]]
            off_events = off_events[on_duration >= self.min_on[0]]
            assert len(on_events) == len(off_events)

        temp_status = np.zeros_like(data)
        for on, off in zip(on_events, off_events):
            temp_status[on:off] = 1
        return temp_status

    # 保留原有的 get_train_datasets 等方法（如果其他地方用到）
    def get_train_datasets(self):
        val_end = int(self.val_size * len(self.x))
        val = NILMDataset(self.x[:val_end], self.y[:val_end], self.status[:val_end],
                          self.window_size, self.window_size)
        train = NILMDataset(self.x[val_end:], self.y[val_end:], self.status[val_end:],
                            self.window_size, self.window_stride)
        return train, val

    def get_pretrain_datasets(self, mask_prob=0.25):
        val_end = int(self.val_size * len(self.x))
        val = NILMDataset(self.x[:val_end], self.y[:val_end], self.status[:val_end],
                          self.window_size, self.window_size)
        train = Pretrain_Dataset(self.x[val_end:], self.y[val_end:], self.status[val_end:],
                                 self.window_size, self.window_stride, mask_prob=mask_prob)
        return train, val