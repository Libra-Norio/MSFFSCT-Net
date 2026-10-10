import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from NILM_Dataset import NILMDataset
from Pretrain_Dataset import Pretrain_Dataset


class Redd_Parser:
    def __init__(self, args, stats=None):
        self.data_location   = args.redd_location
        self.house_indicies  = args.house_indicies
        self.appliance_names = args.appliance_names
        self.sampling        = args.sampling
        self.normalize       = args.normalize

        # ★★★ 处理 REDD 拼写错误：将 dishwaser 映射到 dishwasher 读取参数 ★★★
        def get_param(key):
            if key == 'dishwaser':
                key = 'dishwasher'
            return args.cutoff[key]

        self.cutoff          = [get_param(appl) for appl in ['aggregate'] + self.appliance_names]
        self.threshold       = [args.threshold[appl.replace('dishwaser', 'dishwasher')] for appl in args.appliance_names]
        self.min_on          = [args.min_on[appl.replace('dishwaser', 'dishwasher')] for appl in args.appliance_names]
        self.min_off         = [args.min_off[appl.replace('dishwaser', 'dishwasher')] for appl in args.appliance_names]

        self.val_size        = args.validation_size
        self.window_size     = args.window_size
        self.window_stride   = args.window_stride

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
        all_dfs = []
        directory = Path(self.data_location)

        for house_id in self.house_indicies:
            house_folder = directory / f'house_{house_id}'
            if not house_folder.exists():
                print(f"警告: 房屋 {house_id} 文件夹不存在，跳过")
                continue

            # 读取标签
            labels = pd.read_csv(house_folder / 'labels.dat', sep=' ', header=None)
            labels.columns = ['channel', 'appliance']
            label_dict = dict(zip(labels['appliance'], labels['channel']))

            # 读取总功率 (channel_1 + channel_2)
            try:
                ch1 = pd.read_csv(house_folder / 'channel_1.dat', sep=' ', header=None)
                ch2 = pd.read_csv(house_folder / 'channel_2.dat', sep=' ', header=None)
                main = pd.merge(ch1, ch2, on=0, how='inner')
                main.iloc[:, 1] = main.iloc[:, 1] + main.iloc[:, 2]
                main = main.iloc[:, [0, 1]]
                main.columns = [0, 'aggregate']
            except FileNotFoundError:
                print(f"警告: 房屋 {house_id} 缺少总功率通道，跳过")
                continue

            # 提取目标设备数据
            device_dfs = []
            for app in self.appliance_names:
                # 标签中的名称可能与 app 不完全一致，但 dishwaser 特殊
                if app in label_dict:
                    ch = label_dict[app]
                else:
                    # 如果 app 是 dishwaser，尝试匹配 dishwaser（实际标签）
                    if app == 'dishwaser' and 'dishwaser' in label_dict:
                        ch = label_dict['dishwaser']
                    else:
                        print(f"警告: 房屋 {house_id} 中没有设备 {app}")
                        continue
                ch_path = house_folder / f'channel_{ch}.dat'
                if ch_path.exists():
                    temp = pd.read_csv(ch_path, sep=' ', header=None)
                    if temp.shape[1] >= 2:
                        temp.columns = [0, app]
                        temp = temp[[0, app]]
                        device_dfs.append(temp)
                    else:
                        print(f"警告: 设备 {app} 通道 {ch} 数据格式异常，跳过")
                else:
                    print(f"警告: 房屋 {house_id} 设备 {app} 通道文件不存在")

            if not device_dfs:
                print(f"警告: 房屋 {house_id} 无目标设备数据，跳过")
                continue

            # 合并设备数据
            device_data = device_dfs[0]
            for df in device_dfs[1:]:
                device_data = pd.merge(device_data, df, on=0, how='outer')

            # 合并总功率和设备数据
            merged = pd.merge(main, device_data, on=0, how='inner')
            if merged.empty:
                print(f"警告: 房屋 {house_id} 合并后无数据，跳过")
                continue

            # 设置时间索引
            merged[0] = pd.to_datetime(merged[0], unit='s')
            merged = merged.set_index(0)
            merged = merged.resample(self.sampling).mean().fillna(method='ffill', limit=30)
            merged = merged.dropna()
            if merged.empty:
                print(f"警告: 房屋 {house_id} 重采样后无数据，跳过")
                continue

            # 过滤
            merged = merged[merged['aggregate'] > 0]
            merged[merged < 5] = 0
            # 截断：使用第一个设备的 cutoff（所有设备共享同一个 cutoff）
            for col in ['aggregate'] + self.appliance_names:
                if col in merged.columns:
                    merged[col] = merged[col].clip(upper=self.cutoff[0])

            all_dfs.append(merged)
            print(f"房屋 {house_id} 加载成功，样本数: {len(merged)}")

        if not all_dfs:
            raise ValueError("未加载到任何 REDD 房屋数据")

        entire_data = pd.concat(all_dfs, ignore_index=False)
        entire_data = entire_data.dropna().copy()
        entire_data = entire_data[entire_data['aggregate'] > 0]
        entire_data[entire_data < 5] = 0

        x = entire_data['aggregate'].values.astype(np.float32)
        y = entire_data[self.appliance_names[0]].values.astype(np.float32)
        # 计算状态（但训练时可能不用）
        status = self.compute_status(y)

        return x, y, status

    def compute_status(self, data):
        # 这个函数虽然定义了，但训练时可能不调用，保留以防万一
        # 注意：如果训练时使用状态，需要确保 threshold 正确映射
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

    def get_train_datasets(self):
        val_end = int(self.val_size * len(self.x))
        train = NILMDataset(self.x[val_end:], self.y[val_end:], self.status[val_end:],
                            self.window_size, self.window_stride)
        val = NILMDataset(self.x[:val_end], self.y[:val_end], self.status[:val_end],
                          self.window_size, self.window_size)
        return train, val

    def get_pretrain_datasets(self, mask_prob=0.25):
        val_end = int(self.val_size * len(self.x))
        val = NILMDataset(self.x[:val_end], self.y[:val_end], self.status[:val_end],
                          self.window_size, self.window_size)
        train = Pretrain_Dataset(self.x[val_end:], self.y[val_end:], self.status[val_end:],
                                 self.window_size, self.window_stride, mask_prob=mask_prob)
        return train, val