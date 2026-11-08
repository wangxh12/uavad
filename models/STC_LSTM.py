import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=configs.enc_in,
            hidden_size=128,
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            input_size=128,
            hidden_size=64,
            batch_first=True
        )

        self.lstm3 = nn.LSTM(
            input_size=64,
            hidden_size=32,
            batch_first=True
        )

        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x_enc, x_mark_enc=None,
                x_dec=None, x_mark_dec=None, mask=None):

        # x_enc: [B, D, C]
        x, _ = self.lstm1(x_enc)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)

        # 最后一个时间步
        x = x[:, -1, :]

        x = self.fc1(x)
        x = self.fc2(x)

        return x.unsqueeze(1)  # [B, 1, 1]