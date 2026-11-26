from bisect import bisect_right
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset


class ALFAForecast(Dataset):
    """Forecasting-based anomaly detection on merged multi-flight sequences."""

    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="M",
        target="field.measured.roll",
        scaler="standard",
        stride=1,
        timeenc=0,
        freq="h",
        args=None,
    ):
        self.seq_len, self.label_len, self.pred_len = size
        self.flag = flag
        self.features = features
        self.target = target
        self.stride = stride
        self.args = args
        self.root_path = root_path

        if scaler == "standard":
            self.scaler = StandardScaler()
        elif scaler == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

        self.__read_data__()

    def __read_data__(self):
        train_df = pd.read_csv(os.path.join(self.root_path, "train.csv"))

        df_raw = (
            train_df
            if self.flag == "train"
            else pd.read_csv(os.path.join(self.root_path, f"{self.flag}.csv"))
        )

        # 排除非特征字段，并将预测目标放在最后一列
        cols = [
            c
            for c in train_df.columns
            if c not in ("time", "date", "label", self.target)
        ] + [self.target]

        if self.features == "S":
            cols = [self.target]

        train_data = train_df[cols].to_numpy(dtype=np.float32)
        data = df_raw[cols].to_numpy(dtype=np.float32)

        # 仅使用训练集拟合 scaler
        if self.scaler is not None:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        # 输入不包含预测目标（S 模式除外）
        if self.features == "S":
            self.data_x = data
            self.data_y = data
        else:
            self.data_x = data[:, :-1]
            self.data_y = data[:, -1:]

        # 加载异常标签，不参与标准化
        self.data_label = df_raw["label"].to_numpy(dtype=np.int64)

    def __getitem__(self, index):
        s_begin = index * self.stride
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_label = self.data_label[s_end:r_end]

        return seq_x, seq_y, seq_label

    def __len__(self):
        return max(
            0, (len(self.data_x) - self.seq_len - self.pred_len) // self.stride + 1
        )

    def inverse_transform(self, data):
        if self.scaler is not None:
            return self.scaler.inverse_transform(data)
        return data


class ALFARecon(Dataset):
    """Reconstruction-based anomaly detection on merged multi-flight sequences."""

    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="M",
        target="field.measured.roll",
        scaler="standard",
        stride=1,
        timeenc=0,
        freq="h",
        args=None,
    ):

        assert flag in ["train", "val", "test"]
        self.flag = flag

        # Reconstruction only requires seq_len.
        self.seq_len = size[0] if size else 9
        # self.stride = stride
        self.stride = self.seq_len if flag == "test" else stride

        self.features = features
        self.target = target
        self.root_path = root_path
        self.args = args

        if scaler == "standard":
            self.scaler = StandardScaler()
        elif scaler == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

        self.__read_data__()

    def __read_data__(self):
        files = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}

        train_df = pd.read_csv(os.path.join(self.root_path, files["train"]))

        if self.flag == "train":
            df_raw = train_df
        else:
            df_raw = pd.read_csv(os.path.join(self.root_path, files[self.flag]))

        # Exclude timestamp and anomaly label.
        exclude_cols = ["date", "label"]

        feature_cols = [c for c in train_df.columns if c not in exclude_cols]

        if not set(feature_cols).issubset(df_raw.columns):
            raise ValueError("Inconsistent feature columns")

        # Select input features.
        if self.features in ["M", "MS"]:
            cols = feature_cols
        elif self.features == "S":
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
        labels = df_raw["label"].to_numpy()
        labels = (labels != 0).astype(np.int64)

        self.data_x = data
        self.data_y = labels

    def __getitem__(self, index):
        s_begin = index * self.stride
        s_end = s_begin + self.seq_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_begin:s_end]

        return seq_x, seq_y

    def __len__(self):
        return max(0, (len(self.data_x) - self.seq_len) // self.stride + 1)

    def inverse_transform(self, data):
        if self.scaler is None:
            return data
        return self.scaler.inverse_transform(data)


class ALFASegForecast(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="MS",
        target="field.measured.roll",
        scaler="standard",
        timeenc=0,
        freq="h",
        stride=1,
        args=None,
    ):
        # size: [seq_len, label_len, pred_len]
        self.seq_len, self.label_len, self.pred_len = size

        assert features == "MS"
        assert stride == 1 and self.pred_len == 1
        assert self.label_len == 0

        self.root_path = Path(root_path)
        self.flag = flag
        self.features = features
        self.target = target
        self.stride = stride

        if scaler == "standard":
            self.scaler = StandardScaler()
        elif scaler == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

        self.__read_data__()

    def __read_data__(self):
        files = sorted((self.root_path / self.flag).glob("*.csv"))
        frames = [pd.read_csv(file) for file in files]

        # Exclude time and label; move the target to the last column.
        cols = [
            c for c in frames[0].columns if c not in ("time", "label", self.target)
        ] + [self.target]

        # Fit the scaler on training data only.
        if self.scaler is not None:
            if self.flag == "train":
                train_frames = frames
            else:
                train_files = sorted((self.root_path / "train").glob("*.csv"))
                train_frames = (pd.read_csv(file) for file in train_files)

            for df in train_frames:
                self.scaler.partial_fit(df[cols].to_numpy(dtype=np.float64))

        # Store each flight independently.
        self.data = []
        self.labels = []
        self.file_names = []
        self.window_sizes = []

        for file, df in zip(files, frames):
            n_windows = max(
                0, (len(df) - self.seq_len - self.pred_len) // self.stride + 1
            )

            if n_windows == 0:
                continue

            data = df[cols].to_numpy(dtype=np.float64)

            if self.scaler is not None:
                data = self.scaler.transform(data)

            self.data.append(data.astype(np.float32))
            self.labels.append(df["label"].to_numpy(dtype=np.int64))
            self.file_names.append(file.name)
            self.window_sizes.append(n_windows)

        self.cumulative_sizes = np.cumsum([0] + self.window_sizes)

        self.enc_in = len(cols) - 1
        self.c_out = 1

    def __getitem__(self, index):
        file_idx = bisect_right(self.cumulative_sizes, index) - 1

        local_idx = index - self.cumulative_sizes[file_idx]

        s_begin = local_idx * self.stride
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len

        data = self.data[file_idx]
        labels = self.labels[file_idx]

        seq_x = data[s_begin:s_end, :-1]
        seq_y = data[s_end:r_end, -1:]
        seq_label = labels[s_end:r_end]

        return seq_x, seq_y, seq_label

    def __len__(self):
        return int(self.cumulative_sizes[-1])


class ALFASegRecon(Dataset):
    """Reconstruction windows from independent flights in train/val/test folders."""

    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="M",
        target="field.measured.roll",
        scaler="standard",
        stride=1,
        timeenc=0,
        freq="h",
        args=None,
    ):
        self.root_path = Path(root_path)
        self.flag = flag
        self.seq_len = size[0] if size else 9
        self.features = features
        self.target = target
        self.stride = stride
        self.args = args

        if scaler == "standard":
            self.scaler = StandardScaler()
        elif scaler == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

        self.__read_data__()

    def __read_data__(self):
        files = sorted((self.root_path / self.flag).glob("*.csv"))
        train_files = sorted((self.root_path / "train").glob("*.csv"))

        first_df = pd.read_csv(files[0])
        feature_cols = [
            column
            for column in first_df.columns
            if column not in ("time", "date", "label")
        ]
        self.feature_cols = [self.target] if self.features == "S" else feature_cols

        if self.scaler is not None:
            for file in train_files:
                train_df = pd.read_csv(file)
                self.scaler.partial_fit(
                    train_df[self.feature_cols].to_numpy(dtype=np.float64)
                )

        self.data = []
        self.labels = []
        self.file_names = []
        self.window_sizes = []
        for file in files:
            df = pd.read_csv(file)
            n_windows = max(0, (len(df) - self.seq_len) // self.stride + 1)
            if n_windows == 0:
                continue

            values = df[self.feature_cols].to_numpy(dtype=np.float64)
            if self.scaler is not None:
                values = self.scaler.transform(values)

            self.data.append(values.astype(np.float32))
            self.labels.append((df["label"].to_numpy() != 0).astype(np.int64))
            self.file_names.append(file.name)
            self.window_sizes.append(n_windows)

        self.cumulative_sizes = np.cumsum([0] + self.window_sizes)
        self.enc_in = len(self.feature_cols)
        self.c_out = self.enc_in

    def __getitem__(self, index):
        file_idx = bisect_right(self.cumulative_sizes, index) - 1
        local_idx = index - self.cumulative_sizes[file_idx]
        start = local_idx * self.stride
        end = start + self.seq_len
        return self.data[file_idx][start:end], self.labels[file_idx][start:end]

    def __len__(self):
        return int(self.cumulative_sizes[-1])

    def inverse_transform(self, data):
        if self.scaler is None:
            return data
        return self.scaler.inverse_transform(data)
