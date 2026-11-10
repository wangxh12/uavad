"""FACT adapted for ALFA cross-variable, single-target forecasting.

Source architecture: FACT (ICLR 2026), Wang et al.
Official implementation: https://github.com/wanghq21/FACT

Task contract:
    x_enc: [B, seq_len, enc_in] -- ONLY non-target sensor variables
    output: [B, pred_len, 1] -- a separately selected target variable

The time-domain and amplitude/phase-domain DConvBlocks are retained.
Compared with the official forecasting head, a final cross-variable
projection is added because the target variable is absent from the input.
This is a task-specific adaptation, not an exact benchmark reproduction.
"""

import math

import torch
import torch.nn as nn


class Inception2d(nn.Module):
    """Multi-kernel depth-wise 2D convolutions, as used in FACT."""

    def __init__(self, channels, num_kernels=3, dilation=1):
        super().__init__()
        if num_kernels < 2:
            raise ValueError("num_kernels must be >= 2 for variable interaction")
        self.kernels = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=1, groups=channels)]
            + [
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=2 * k + 1,
                    padding=dilation * k,
                    dilation=dilation,
                    groups=channels,
                )
                for k in range(1, num_kernels)
            ]
        )
        for layer in self.kernels:
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        out = self.kernels[0](x)
        for layer in self.kernels[1:]:
            out = out + layer(x)
        return out / len(self.kernels)


class Conv_block2d(nn.Module):
    """FACT DConvBlock on a 2D arrangement of the input sensor variables."""

    def __init__(self, enc_in, d_model, d_ff, dilation=2, num_kernels=3, dropout=0.1):
        super().__init__()
        self.enc_in = int(enc_in)

        # Minimal square layout; grid positions beyond enc_in repeat variables
        # cyclically, mirroring the original model's intended padding scheme.
        self.grid_size = math.isqrt(self.enc_in)
        if self.grid_size * self.grid_size < self.enc_in:
            self.grid_size += 1
        grid_count = self.grid_size ** 2
        self.register_buffer(
            "grid_index", torch.arange(grid_count) % self.enc_in,
            persistent=False,
        )

        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(d_model),
                Inception2d(d_model, num_kernels=num_kernels, dilation=k),
            )
            for k in range(1, dilation + 1)
        ])
        self.conv_ffn = nn.Sequential(
            nn.BatchNorm2d(d_model),
            nn.Conv2d(d_model, d_ff, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(d_ff, d_model, kernel_size=1),
        )

    def forward(self, x):
        # x: [B, embedding_channels, num_sensor_variables]
        b, d, c = x.shape
        if c != self.enc_in:
            raise ValueError(f"DConvBlock expected {self.enc_in} variables, got {c}")
        grid = x.index_select(dim=2, index=self.grid_index)
        grid = grid.reshape(b, d, self.grid_size, self.grid_size)

        out = grid
        for conv in self.conv:
            out = out + conv(grid)  # parallel dilations + residual
        out = out + self.conv_ffn(out)
        return out.flatten(2)[:, :, :c]


def _parse_dilation(value):
    """Accept [1, 2], (1, 2), '1,2', '[1,2]', or an integer."""
    if isinstance(value, str):
        value = value.strip().strip("[]()")
        value = [v for v in value.replace(",", " ").split() if v]
    elif isinstance(value, int):
        value = [value]
    values = [int(v) for v in value]
    if not values or any(v < 1 for v in values):
        raise ValueError("dilation needs one or more positive integers")
    return values


class Model(nn.Module):
    """TSLib-style FACT model for a target absent from x_enc."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = getattr(configs, "task_name", "forecast")
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)  # excludes the target channel
        self.c_out = int(getattr(configs, "c_out", 1))
        self.d_model = int(getattr(configs, "d_model", 32))
        self.d_ff = int(getattr(configs, "d_ff", 64))
        self.dropout = float(getattr(configs, "dropout", 0.1))
        self.num_kernels = int(getattr(configs, "num_kernels", 3))
        self.dilation = _parse_dilation(getattr(configs, "dilation", [1, 2]))
        self.alpha = float(getattr(configs, "core", 0.5))
        self.use_norm = bool(getattr(configs, "use_norm", False))


        # Original FACT: project the history-length dimension into latent time
        # features while leaving the variable dimension intact.
        self.emb = nn.Conv1d(self.seq_len, self.d_model, kernel_size=1)

        def make_stack(features):
            return nn.ModuleList([
                Conv_block2d(
                    enc_in=self.enc_in,
                    d_model=features,
                    d_ff=self.d_ff,
                    dilation=d,
                    num_kernels=self.num_kernels,
                    dropout=self.dropout,
                )
                for d in self.dilation
            ])

        if self.alpha > 0.0:
            self.conv_time = make_stack(self.d_model)
        if self.alpha < 1.0:
            self.conv_amp = make_stack(self.d_model // 2 + 1)
            self.conv_phase = make_stack(self.d_model // 2 + 1)

        # Original FACT per-variable temporal forecast head.
        self.projection = nn.Conv1d(self.d_model, self.pred_len, kernel_size=1)
        # ADAPTATION: map *other* sensors' forecasts to the withheld target.
        self.target_head = nn.Linear(self.enc_in, self.c_out)

    def _time_branch(self, x):
        for layer in self.conv_time:
            x = layer(x)
        return x

    def _frequency_branch(self, x):
        # FFT is taken over FACT's latent temporal-feature dimension.
        # float() keeps FFT compatible with mixed-precision training.
        spectrum = torch.fft.rfft(x.float(), dim=1)
        amplitude = torch.abs(spectrum)
        phase = torch.angle(spectrum)
        for layer in self.conv_amp:
            amplitude = layer(amplitude)
        for layer in self.conv_phase:
            phase = layer(phase)

        # Autocast convolution can return fp16/bf16; complex FFT reconstruction
        # is explicitly carried out in fp32 for arbitrary d_model values.
        amplitude = amplitude.float()
        phase = phase.float()

        # Preserve FACT's amplitude/phase reconstruction semantics, including
        # the possibility of negative learned amplitude responses.
        phase = phase - 2 * math.pi * torch.round(phase / (2 * math.pi))
        spectrum = torch.complex(
            amplitude * torch.cos(phase),
            amplitude * torch.sin(phase),
        )
        return torch.fft.irfft(spectrum, n=self.d_model, dim=1)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if self.use_norm:
            mean = x_enc.mean(dim=1, keepdim=True).detach()
            scale = torch.sqrt(
                x_enc.var(dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = (x_enc - mean) / scale
            # No reverse normalization: the held-out target has NO observed
            # history from which to derive its instance statistics.

        hidden = self.emb(x_enc)  # [B, d_model, enc_in]
        if self.alpha == 1.0:
            hidden = self._time_branch(hidden)
        elif self.alpha == 0.0:
            hidden = self._frequency_branch(hidden)
        else:
            time_hidden = self._time_branch(hidden)
            freq_hidden = self._frequency_branch(hidden)
            hidden = self.alpha * time_hidden + (1 - self.alpha) * freq_hidden

        per_sensor_pred = self.projection(hidden)  # [B, pred_len, enc_in]
        target_pred = self.target_head(per_sensor_pred)  # [B, pred_len, 1]
        return target_pred

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == "forecast":
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        return None
