from bisect import bisect_right
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import Dataset
from utils.timefeatures import time_features


class Dataset_ALFA_Forecast(Dataset):
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
        else:
            self.data_x = data[:, :-1]

        self.data_y = data

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


class Dataset_ALFA_Forecast_MultiFlight(Dataset):
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


class Dataset_ALFA_Reconstruction(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="M",
        target="field.measured.roll",
        scaler="standard",
        timeenc=0,
        freq="h",
        stride=1,
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
        self.timeenc = timeenc
        self.freq = freq
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


class PreTrainSegLoader(Dataset):
    def __init__(self, root_path, seq_len, pred_len, stride, flag="train"):
        self.seq_len = seq_len
        self.data_list = []
        self.label_list = []
        self.starts_list = []
        self.n_window_list = [0]

        for path in sorted((Path(root_path) / flag).glob("*.csv")):
            frame = pd.read_csv(path)
            # Select input features and labels.
            data = frame.iloc[:, 1:-1].to_numpy(dtype=np.float32)
            labels = frame.iloc[:, -1].to_numpy(dtype=np.float32)[:, None]
            starts = np.arange(0, len(frame) - seq_len + 1, stride)

            if len(starts) == 0:
                continue

            self.data_list.append(data)
            self.label_list.append(labels)
            self.starts_list.append(starts)
            self.n_window_list.append(
                self.n_window_list[-1] + len(starts) * data.shape[1]
            )

        self.n_window_list = np.asarray(self.n_window_list)

    def __len__(self):
        return int(self.n_window_list[-1])

    def __getitem__(self, index):
        file_id = np.searchsorted(self.n_window_list, index, side="right") - 1

        local_index = index - self.n_window_list[file_id]
        channel, window = divmod(local_index, len(self.starts_list[file_id]))

        start = self.starts_list[file_id][window]
        end = start + self.seq_len

        seq_x = self.data_list[file_id][start:end, channel : channel + 1]
        lab_y = self.label_list[file_id][start:end]

        return seq_x, lab_y


class EvalSegLoader(Dataset):
    def __init__(self, data_path, seq_len, flag="test"):
        self.data = []
        self.labels = []
        self.flight_lengths = []

        for path in sorted((Path(data_path) / flag).glob("*.csv")):
            frame = pd.read_csv(path)
            windows = len(frame) // seq_len

            if windows == 0:
                continue

            end = windows * seq_len
            self.data.append(
                frame.iloc[:end, 1:-1]
                .to_numpy(np.float32)
                .reshape(windows, seq_len, -1)
            )
            self.labels.append(
                frame.iloc[:end, -1].to_numpy(np.int64).reshape(windows, seq_len)
            )
            self.flight_lengths.append(end)

        self.data = np.concatenate(self.data)
        self.labels = np.concatenate(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class ALFAForecastDataset(Dataset):
    """Return (seq_x, lab_x, seq_y, lab_y) without crossing flight boundaries."""

    def __init__(
        self,
        root_path: str | Path,
        flag: str,
        seq_len: int = 100,
        pred_len: int = 1,
        stride: int = 1,
    ) -> None:
        assert flag in ("train", "val", "test")

        self.root_path = Path(root_path)
        self.flag = flag
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride

        with (self.root_path / "manifest.json").open(encoding="utf-8") as f:
            meta = json.load(f)
        self.feature_columns = meta["features"]
        self.flight_names = meta["splits"][flag]

        # Fit ONLY on the six normal training flights, including when loading val/test.
        train_data = [
            pd.read_csv(self.root_path / "train" / name)[
                self.feature_columns
            ].to_numpy()
            for name in meta["splits"]["train"]
        ]
        self.scaler = StandardScaler().fit(np.concatenate(train_data, axis=0))

        self.series = []
        self.labels = []
        self.times = []
        self.window_index = []  # (flight_idx, start_idx) -- no cross-flight window
        for flight_idx, name in enumerate(self.flight_names):
            df = pd.read_csv(self.root_path / flag / name)
            values = df[self.feature_columns].to_numpy()
            labels = df["label"].to_numpy(dtype=np.float32).reshape(-1, 1)
            n = len(df)

            if n < seq_len + pred_len:
                raise ValueError(f"架次 {name} 过短：{n} < {seq_len + pred_len}")

            self.series.append(self.scaler.transform(values).astype(np.float32))
            self.labels.append(labels)
            self.times.append(df["time"].to_numpy())
            self.window_index.extend(
                (flight_idx, start)
                for start in range(0, n - seq_len - pred_len + 1, stride)
            )

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, index: int):
        flight_idx, start = self.window_index[index]
        s_end = start + self.seq_len
        r_end = s_end + self.pred_len
        x = self.series[flight_idx]
        y = self.labels[flight_idx]

        return (
            x[start:s_end],
            y[start:s_end],
            x[s_end:r_end],
            y[s_end:r_end],
        )

    def window_info(self, index: int) -> dict:
        """Map predictions back to an original flight and its time indices."""
        flight_idx, start = self.window_index[index]
        target_start = start + self.seq_len
        target_end = target_start + self.pred_len
        return {
            "flight": self.flight_names[flight_idx],
            "input_start": start,
            "target_start": target_start,
            "target_end": target_end,
            "target_times": self.times[flight_idx][target_start:target_end].copy(),
        }
