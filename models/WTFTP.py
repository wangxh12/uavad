"""WTFTP (9 -> 1) adapted to the Time-Series-Library model interface.

Architecture follows the published WTFTP design: input embedding, an LSTM
encoder, two wavelet-attention/decoder branches, supervised Haar coefficients,
and an inverse discrete wavelet transform.

Reference: https://github.com/MusDev7/wtftp-model (Apache-2.0).
This implementation is specifically for the paper's default 1-level Haar DWT.
No additional wavelet package is required.
"""

import math

import torch
from torch import nn


class _TemporalGating(nn.Module):
    """Learn attention-like weights over encoder time steps."""

    def __init__(self, steps: int, hidden_dim: int, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.compress = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim // 2, 1), bias=False),
            nn.ReLU(),
            nn.Linear(max(hidden_dim // 2, 1), 1, bias=False),
        )
        self.activate = nn.Sequential(
            nn.Linear(steps, max(steps // 2, 1), bias=False),
            nn.ReLU(),
            nn.Linear(max(steps // 2, 1), steps, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, encoded: torch.Tensor):
        # encoded: [B, T, H]; weights: [B, 1, T]
        importance = self.compress(encoded).transpose(1, 2)
        weights = self.activate(importance)
        gated = encoded * weights.transpose(1, 2)
        if self.skip:
            gated = gated + encoded
        return gated, weights


class _WaveletAttention(nn.Module):
    """Weight encoder states and align them with length-5 DWT coefficients."""

    def __init__(self, steps: int, hidden_dim: int):
        super().__init__()
        self.gate = _TemporalGating(steps, hidden_dim)
        # The original WTFTP level-1 attention uses kernel=2, stride=2,
        # padding=1. For T=9 this produces exactly 5 coefficient steps.
        self.align = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2,
                      stride=2, padding=1, bias=False),
            nn.ReLU(),
        )

    def forward(self, encoded: torch.Tensor):
        weighted, weights = self.gate(encoded)
        coefficients_input = self.align(weighted.transpose(1, 2)).transpose(1, 2)
        return coefficients_input, weights


class Model(nn.Module):
    """TSLib-compatible WTFTP single-step forecasting model.

    Required args: seq_len=9, pred_len=1, enc_in=input channels, c_out=target channels.
    Optional args: d_model=64, e_layers=4, d_layers=1, dropout=0.0.

    forward(x)                        -> [B, 1, c_out]
    forward(x, return_aux=True)       -> dict with forecast, reconstructed
                                       10-step sequence, and DWT coefficients
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.c_out = int(getattr(configs, 'c_out', self.enc_in))
        d_model = int(getattr(configs, 'd_model', 64))
        e_layers = int(getattr(configs, 'e_layers', 4))
        d_layers = int(getattr(configs, 'd_layers', 1))
        dropout = float(getattr(configs, 'dropout', 0.0))

        if (self.seq_len, self.pred_len) != (9, 1):
            raise ValueError('This WTFTP adapter implements 9 -> 1 only. '
                             'Set --seq_len 9 --pred_len 1.')
        if self.enc_in < 1 or self.c_out < 1 or d_model < 2 or e_layers < 1 or d_layers < 1:
            raise ValueError('Positive channel/layer counts and d_model >= 2 are required.')

        self.hidden_dim = d_model
        self.decoder_layers = d_layers

        self.embedding = nn.Sequential(
            nn.Linear(self.enc_in, d_model // 2, bias=False),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model, bias=False),
            nn.ReLU(),
        )
        self.encoder = nn.LSTM(
            input_size=d_model, hidden_size=d_model, num_layers=e_layers,
            batch_first=True, bias=False,
            dropout=dropout if e_layers > 1 else 0.0,
        )
        # Detail (high-frequency) and approximation (low-frequency) branches.
        self.detail_attention = _WaveletAttention(self.seq_len, d_model)
        self.approx_attention = _WaveletAttention(self.seq_len, d_model)
        self.detail_decoder = nn.LSTM(
            d_model, d_model, num_layers=d_layers, batch_first=True,
            bias=False, dropout=dropout if d_layers > 1 else 0.0,
        )
        self.approx_decoder = nn.LSTM(
            d_model, d_model, num_layers=d_layers, batch_first=True,
            bias=False, dropout=dropout if d_layers > 1 else 0.0,
        )
        self.detail_norm = nn.LayerNorm(d_model)
        self.approx_norm = nn.LayerNorm(d_model)
        self.detail_out = nn.Linear(d_model, self.c_out)
        self.approx_out = nn.Linear(d_model, self.c_out)

    @staticmethod
    def haar_forward(sequence: torch.Tensor):
        """Orthogonal Haar DWT of an even-length [B, T, C] sequence.

        Returns (lo, hi), both [B, T/2, C]. No future data is used by
        the model; the experiment calls this on the *target* for supervision.
        """
        if sequence.ndim != 3 or sequence.size(1) % 2 != 0:
            raise ValueError('Haar DWT expects [B, even_T, C].')
        even, odd = sequence[:, ::2, :], sequence[:, 1::2, :]
        return (even + odd) / math.sqrt(2.0), (even - odd) / math.sqrt(2.0)

    @staticmethod
    def haar_inverse(lo: torch.Tensor, hi: torch.Tensor):
        """Differentiable IDWT: [B, 5, C] * 2 -> [B, 10, C]."""
        if lo.shape != hi.shape:
            raise ValueError('Low/high Haar coefficient shapes do not match.')
        even = (lo + hi) / math.sqrt(2.0)
        odd = (lo - hi) / math.sqrt(2.0)
        return torch.stack([even, odd], dim=2).flatten(1, 2)

    def _decode(self, attention, decoder, norm, projection, encoded, encoder_cell):
        features, weights = attention(encoded)
        B = features.size(0)
        initial_hidden = features.new_zeros(self.decoder_layers, B, self.hidden_dim)
        if self.decoder_layers == 1:
            initial_cell = encoder_cell[-1:]
        else:
            remaining = features.new_zeros(self.decoder_layers - 1, B, self.hidden_dim)
            initial_cell = torch.cat([encoder_cell[-1:], remaining], dim=0)
        hidden_seq, _ = decoder(features, (initial_hidden, initial_cell))
        coeff = projection(norm(hidden_seq))
        return coeff, weights

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                return_aux=False):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f'Expected [B, {self.seq_len}, {self.enc_in}], '
                             f'got {tuple(x_enc.shape)}.')
        embedded = self.embedding(x_enc)
        encoded, (_, cell) = self.encoder(embedded)
        hi, high_weights = self._decode(
            self.detail_attention, self.detail_decoder,
            self.detail_norm, self.detail_out, encoded, cell,
        )
        lo, low_weights = self._decode(
            self.approx_attention, self.approx_decoder,
            self.approx_norm, self.approx_out, encoded, cell,
        )
        reconstruction = self.haar_inverse(lo, hi)
        if reconstruction.size(1) != self.seq_len + self.pred_len:
            raise RuntimeError('Wavelet/sequence length mismatch.')
        forecast = reconstruction[:, -self.pred_len:, :]
        if not return_aux:
            return forecast
        return {
            'forecast': forecast,
            'reconstruction': reconstruction,
            'low_coeff': lo,
            'high_coeff': hi,
            'attention_low': low_weights,
            'attention_high': high_weights,
        }
