import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import Dataset
from utils.timefeatures import time_features


class Dataset_ALFA_Forecast(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='MS',
                 target='field.measured.roll', scale='standard', timeenc=0, freq='h', step=1, args=None):
        # size [seq_len, label_len, pred_len]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.seq_len, self.label_len, self.pred_len = size or [9, 0, 1]
        assert step == 1 and self.pred_len == 1
        
        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq
        self.step = step
        self.args = args
        
        if scale == 'standard':
            self.scaler = StandardScaler()
        elif scale == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

        self.root_path = root_path
        self.__read_data__()
        
    def __read_data__(self):
        files = ['train.csv', 'val.csv', 'test.csv']
        frames = [pd.read_csv(os.path.join(self.root_path, f)) for f in files]
        train_df, val_df, test_df = frames
        
        cols = list(train_df.columns)
        assert set(cols) == set(val_df.columns) == set(test_df.columns), \
            'Inconsistent feature columns between train, val and test'

        # Keep numeric variables in train-column order; put target last.
        cols = (['date'] if 'date' in cols else []) + [
            c for c in cols if c not in ('date', self.target)
        ] + [self.target]

        df_raw = pd.concat([df[cols] for df in frames], ignore_index=True)
        n_train, n_val, n_test = map(len, frames)
        border1s = [0, n_train, n_train + n_val]
        border2s = [n_train, n_train + n_val, n_train + n_val + n_test]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw.drop(columns=['date', 'label'], errors='ignore')
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        data = df_data.to_numpy(dtype=np.float32)
        if self.scaler:
            self.scaler.fit(data[:n_train])
            data = self.scaler.transform(data)

        if 'date' in df_raw.columns:
            dates = pd.to_datetime(df_raw['date'])
            stamps = np.column_stack([
                dates.dt.month, dates.dt.day,
                dates.dt.dayofweek, dates.dt.hour,
            ]).astype(np.float32)
            if self.timeenc == 1:
                stamps = stamps / np.array([12., 31., 6., 23.], dtype=np.float32) - 0.5
        else:
            # ALFA files without a timestamp use zero-valued time marks.
            stamps = np.zeros((len(df_raw), 4), dtype=np.float32)
            
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = stamps[border1:border2]
            
    def __getitem__(self, index):
        s_begin = index * self.step
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)
        return max(
            0,
            (len(self.data_x) - self.seq_len - self.pred_len)
            // self.step + 1
        )

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ALFA_Reconstruction(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                features='M', target='field.measured.roll',
                scale='standard', timeenc=0, freq='h', step=1, args=None):
        
        assert flag in ['train', 'val', 'test']
        self.flag = flag

        # Reconstruction only requires seq_len.
        self.seq_len = size[0] if size else 9
        # self.step = step
        self.step = self.seq_len if flag == 'test' else step

        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.args = args

        if scale == 'standard':
            self.scaler = StandardScaler()
        elif scale == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

        self.__read_data__()
        
    def __read_data__(self):
        files = {
            'train': 'train.csv',
            'val': 'val.csv',
            'test': 'test.csv'
        }

        train_df = pd.read_csv(
            os.path.join(self.root_path, files['train'])
        )

        if self.flag == 'train':
            df_raw = train_df
        else:
            df_raw = pd.read_csv(
                os.path.join(self.root_path, files[self.flag])
            )

        # Exclude timestamp and anomaly label.
        exclude_cols = ['date', 'label']

        feature_cols = [
            c for c in train_df.columns
            if c not in exclude_cols
        ]

        if not set(feature_cols).issubset(df_raw.columns):
            raise ValueError("Inconsistent feature columns")

        # Select input features.
        if self.features in ['M', 'MS']:
            cols = feature_cols
        elif self.features == 'S':
            cols = [self.target]
        else:
            raise ValueError(f"Unsupported features: {self.features}")

        train_data = train_df[cols].to_numpy(dtype=np.float32)
        data = df_raw[cols].to_numpy(dtype=np.float32)

        # Fit scaler only on training data.
        if self.scaler is not None:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        # Point-wise anomaly labels.
        labels = df_raw['label'].to_numpy()
        labels = (labels != 0).astype(np.int64)

        self.data_x = data
        self.data_y = labels
        
    def __getitem__(self, index):
        s_begin = index * self.step
        s_end = s_begin + self.seq_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_begin:s_end]

        return seq_x, seq_y

    def __len__(self):
        return max(
            0,
            (len(self.data_x) - self.seq_len) // self.step + 1
        )

    def inverse_transform(self, data):
        if self.scaler is None:
            return data
        return self.scaler.inverse_transform(data)















############################################################# TSLib ###########################################################

class UCRAnomalyloader(Dataset):
    def __init__(self, root_path, data_path, seq_len, patch_len, flag="train"):
        self.root_path = root_path
        self.data_path = data_path
        self.seq_len = seq_len  # 总长度 768
        self.input_len = seq_len - patch_len  # 输入长度，对应7个token = 672
        self.patch_len = patch_len  # 预测长度 token = 96
        self.flag = flag
        self.stride = 1 if self.flag == "train" else self.seq_len - 2 * self.patch_len  # 滑动窗口策略
        # 模式	        步长	                                说明
        # 训练	        stride=1	                        最大化训练样本，密集滑动
        # 测试  stride = seq_len - 2 * patch_len   非重叠窗口，确保每个点只被预测一次
        self.dataset_file_path = os.path.join(self.root_path, self.data_path)
        data_list = []
        assert self.dataset_file_path.endswith('.txt')
        try:
            with open(self.dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    data_list.append(data_line)
            self.data = np.stack(data_list, 0)
        except ValueError:
            with open(self.dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line[0].split()])
            self.data = data_line
            self.data = np.expand_dims(self.data, axis=1)

        self.border = self.find_border_number(self.data_path)
        self.scaler = StandardScaler()
        self.scaler.fit(self.data[:self.border])
        self.data = self.scaler.transform(self.data)
        if self.flag == "train":
            self.data = self.data[:self.border]  # 前self.border个点
        else:
            self.data = self.data[self.border - self.patch_len:]  # 测试集：从训练集最后patch_len开始，包含整个测试集

    def find_border_number(self, input_string):
        parts = input_string.split('_')

        if len(parts) < 3:
            return None

        border_str = parts[-3]

        try:
            border = int(border_str)
            return border
        except ValueError:
            return None

    def __len__(self):
        return (self.data.shape[0] - self.seq_len) // self.stride + 1

    def __getitem__(self, index):
        index = index * self.stride
        return self.data[index:index + self.seq_len, :]




def _read_features(path):
    if not path.is_file():
        raise FileNotFoundError(
            f'Data file not found: {path}. Check --root_path and the split filename.'
        )
    frame = pd.read_csv(path)
    if frame.empty or not frame.columns.is_unique:
        raise ValueError(f'Empty data or duplicate feature columns: {path}')
    try:
        frame = frame.apply(pd.to_numeric, errors='raise')
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Non-numeric features: {path}') from exc
    if not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
        raise ValueError(f'Non-finite features: {path}')
    return frame



class UAVAnomalyDataset(Dataset):
    """Reconstruct every CSV feature and align labels with each window point."""

    def __init__(self, args, flag):
        if flag not in {'train', 'val', 'test'}:
            raise ValueError(f'Unknown split: {flag}')
        if args.features != 'M' or args.c_out != args.enc_in:
            raise ValueError('Reconstruction requires features=M and c_out=enc_in')
        self.seq_len = int(args.seq_len)
        self.step = int(args.step)
        if min(self.seq_len, self.step) < 1:
            raise ValueError('seq_len and step must be positive')

        root = Path(args.root_path)
        train_path = root / args.train_file
        train = _read_features(train_path)
        self.columns = list(train.columns)
        if len(self.columns) != args.enc_in:
            raise ValueError(
                f'{train_path} has {len(self.columns)} feature columns, but '
                f'--enc_in={args.enc_in}. Set --enc_in and --c_out to '
                f'{len(self.columns)} for full-channel reconstruction.'
            )
        self.normalization = args.normalization
        if self.normalization == 'standard':
            self.offset = train.mean(axis=0).to_numpy(dtype=np.float64)
            self.scale = train.std(axis=0, ddof=0).to_numpy(dtype=np.float64)
        elif self.normalization == 'minmax':
            self.offset = train.min(axis=0).to_numpy(dtype=np.float64)
            self.scale = train.max(axis=0).to_numpy(dtype=np.float64) - self.offset
        else:
            raise ValueError(f'Unknown normalization: {self.normalization}')
        self.scale[self.scale < 1e-12] = 1.0

        path = root / {'train': args.train_file, 'val': args.val_file,
                       'test': args.test_file}[flag]
        frame = train if flag == 'train' else _read_features(path)
        if list(frame.columns) != self.columns:
            raise ValueError(f'Feature columns/order differ from {train_path}: {path}')
        self.values = ((frame.to_numpy(dtype=np.float64) - self.offset)
                       / self.scale).astype(np.float32)
        if len(self.values) < self.seq_len:
            raise ValueError(f'{path} needs at least {self.seq_len} rows')

        self.labels = np.zeros(len(self.values), dtype=np.int64)
        if flag == 'test':
            label_path = root / args.label_file
            labels = pd.read_csv(label_path)
            if list(labels.columns) != ['label'] or len(labels) != len(self.values):
                raise ValueError(f'{label_path} must have one label column and one label per test row')
            if labels.isna().any().any() or not labels['label'].isin([0, 1]).all():
                raise ValueError('Labels must be binary (0/1), without missing values')
            self.labels = labels['label'].to_numpy(dtype=np.int64)

    def __len__(self):
        return (len(self.values) - self.seq_len) // self.step + 1

    def __getitem__(self, index):
        start = index * self.step
        end = start + self.seq_len
        return (torch.from_numpy(self.values[start:end]),
                torch.from_numpy(self.labels[start:end]))


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        train_path = os.path.join(root_path, "train.csv")
        test_path = os.path.join(root_path, "test.csv")
        label_path = os.path.join(root_path, "test_label.csv")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_df      = pd.read_csv(train_path)
            test_df       = pd.read_csv(test_path)
            test_label_df = pd.read_csv(label_path)


        data = train_df.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        
        test_data = test_df.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = test_label_df.values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)
        

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])