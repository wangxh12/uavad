"""Target-query forecasting with a SOFTS-style STAR encoder.

The target is not part of the observed input channels. Its learned query token
reads cross-variable context from the remaining sensor histories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class STARBlock(nn.Module):
    """Aggregate channel representations into a core and redistribute it."""

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.core_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def _pool_core(self, x):
        # x: [batch, variables, d_model]
        core_candidates = self.core_projection(x)

        if self.training:
            batch_size, num_variables, d_model = core_candidates.shape
            weights = F.softmax(core_candidates, dim=1)
            weights = weights.transpose(1, 2).reshape(-1, num_variables)
            indices = torch.multinomial(weights, num_samples=1)
            indices = indices.reshape(batch_size, d_model, 1).transpose(1, 2)
            return torch.gather(core_candidates, dim=1, index=indices)

        weights = F.softmax(core_candidates, dim=1)
        return (core_candidates * weights).sum(dim=1, keepdim=True)

    def forward(self, x):
        core = self._pool_core(x).expand(-1, x.size(1), -1)
        fused = self.fusion(torch.cat((x, core), dim=-1))
        x = self.norm1(x + self.dropout(fused))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class Model(nn.Module):
    """Forecast one target from the history of non-target variables.

    Input:  [batch, seq_len, enc_in]
    Output: [batch, pred_len, c_out]
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        self.temporal_embedding = nn.Linear(self.seq_len, self.d_model)
        self.sensor_embedding = nn.Parameter(torch.empty(1, self.enc_in, self.d_model))
        self.target_query = nn.Parameter(torch.empty(1, 1, self.d_model))
        nn.init.normal_(self.sensor_embedding, std=0.02)
        nn.init.normal_(self.target_query, std=0.02)

        self.encoder = nn.ModuleList(
            [
                STARBlock(self.d_model, configs.d_ff, configs.dropout)
                for _ in range(configs.e_layers)
            ]
        )
        self.projection = nn.Linear(self.d_model, self.pred_len * self.c_out)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        sensor_tokens = self.temporal_embedding(x_enc.transpose(1, 2))
        sensor_tokens = sensor_tokens + self.sensor_embedding
        target_query = self.target_query.expand(x_enc.size(0), -1, -1)
        tokens = torch.cat((sensor_tokens, target_query), dim=1)

        for layer in self.encoder:
            tokens = layer(tokens)

        target_state = tokens[:, -1]
        forecast = self.projection(target_state)
        return forecast.reshape(x_enc.size(0), self.pred_len, self.c_out)
