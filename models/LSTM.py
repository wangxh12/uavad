"""Plain multivariate LSTM baseline for next-step roll prediction."""

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.enc_in = int(configs.enc_in)
        self.pred_len = int(configs.pred_len)
        self.c_out = int(configs.c_out)
        hidden = int(configs.d_model)
        layers = int(configs.e_layers)
        dropout = float(configs.dropout)
        if (self.seq_len != 9 or self.pred_len != 1 or self.enc_in != 17
                or self.c_out != 1 or hidden < 1 or layers < 1 or not 0 <= dropout < 1):
            raise ValueError('LSTM baseline expects 9 input steps, 17 channels and one roll target')
        self.encoder = nn.LSTM(self.enc_in, hidden, num_layers=layers,
                               dropout=dropout if layers > 1 else 0.0,
                               batch_first=True)
        self.output = nn.Linear(hidden, self.c_out)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f'Expected [B, {self.seq_len}, {self.enc_in}] input')
        encoded, _ = self.encoder(x_enc)
        return self.output(encoded[:, -1:, :])
