"""TimeCNN forecasting model adapted for the TSLib Model(args) interface.

Reference implementation:
    https://github.com/aohu1105/TimeCNN/blob/main/model/TimeCNN.py

The CrossCNN, FFN, embedding, forecasting head and reversible per-sample
normalization follow the official implementation.

Extension for UAV / target-excluded-from-X forecasting:
    If c_out != enc_in, an extra Linear(enc_in, c_out) projection is applied
    after the original inverse normalization. This extension is NOT part of
    the original TimeCNN paper. Set c_out == enc_in for the original model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    """Official TimeCNN FFN: pre-LN, two pointwise Conv1d, residual, LN."""

    def __init__(self, d_model, d_ff, dropout=0.1, activation="gelu"):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        x = self.norm1(x)  # [B, N, d_model]
        y = self.dropout(self.activation(self.conv1(x.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        return self.norm2(x + y)


class CrossCNN(nn.Module):
    """Timepoint-independent grouped convolution across variables.

    Each of the L time positions gets its own length-N kernel. The cyclic
    concatenation produces N output positions without a circular-pad API.
    """

    def __init__(self, seq_len, c_in, dropout=0.0):
        super().__init__()
        self.crosscnn = nn.Conv1d(
            in_channels=seq_len,
            out_channels=seq_len,
            kernel_size=c_in,
            groups=seq_len,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, N] -> [B, L, 2N-1] -> [B, L, N]
        padded = torch.cat((x[:, :, 1:], x), dim=-1)
        return x + self.dropout(self.crosscnn(padded))


class Model(nn.Module):
    """TimeCNN forecaster compatible with TSLib and simplified forecasting Exps.

    Input:  x_enc of shape [B, seq_len, enc_in].
    Output: forecast of shape [B, pred_len, c_out].

    The original TimeCNN is obtained when c_out == enc_in.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.c_out = int(getattr(configs, "c_out", self.enc_in))
        self.d_model = int(getattr(configs, "d_model", 512))
        d_ff = int(getattr(configs, "d_ff", 2048))
        e_layers = int(getattr(configs, "e_layers", 2))
        dropout = float(getattr(configs, "dropout", 0.1))
        conv_dropout = float(getattr(configs, "conv_dropout", 0.0))
        self.use_norm = bool(getattr(configs, "use_norm", 1))

        if min(self.seq_len, self.pred_len, self.enc_in,
               self.c_out, self.d_model, d_ff) < 1 or e_layers < 0:
            raise ValueError("All dimensions must be positive; e_layers >= 0")

        # All modules below are the original TimeCNN architecture.
        self.crosscnn = CrossCNN(self.seq_len, self.enc_in, conv_dropout)
        self.emb = nn.Linear(self.seq_len, self.d_model)
        self.layers = nn.ModuleList([
            FFN(self.d_model, d_ff, dropout=dropout, activation="gelu")
            for _ in range(e_layers)
        ])
        self.linear_pred = nn.Linear(self.d_model, self.pred_len)

        # Optional target projection: only for models predicting different
        # output variables than those in x_enc (e.g. target_excluded_from_x).
        self.output_projection = (
            nn.Identity() if self.c_out == self.enc_in
            else nn.Linear(self.enc_in, self.c_out)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None,
                x_mark_dec=None, mask=None):
        if x_enc.ndim != 3:
            raise ValueError(f"Expected x_enc [B,L,N], got {tuple(x_enc.shape)}")
        if x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"Expected x_enc [B,{self.seq_len},{self.enc_in}], "
                f"got {tuple(x_enc.shape)}"
            )

        x = x_enc
        if self.use_norm:
            # Exactly the per-instance normalization in official TimeCNN.
            means = x.mean(dim=1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev

        x = self.crosscnn(x)                      # [B, L, N]
        x = self.emb(x.permute(0, 2, 1))        # [B, N, d_model]
        for layer in self.layers:
            x = layer(x)                         # [B, N, d_model]
        x = self.linear_pred(x).permute(0, 2, 1)  # [B, pred_len, N]

        if self.use_norm:
            x = x * stdev + means               # inverse input-variable normalization

        return self.output_projection(x)        # [B, pred_len, c_out]
