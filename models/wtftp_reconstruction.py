"""WTFTP wavelet-attention backbone adapted to same-window reconstruction.

The paper forecasts the next trajectory point from history. Here the complete
96-point, 18-channel window is encoded and reconstructed for anomaly detection.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class WTFTPReconstructionConfig:
    seq_len: int = 96
    channels: int = 18
    embedding: int = 64
    encoder_layers: int = 4
    decoder_layers: int = 1
    dropout: float = 0.0
    wavelet: str = "haar"
    levels: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


def haar_dwt(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthonormal, one-level Haar DWT for even-length [batch, time, channel]."""
    if values.ndim != 3 or values.shape[1] % 2:
        raise ValueError("Haar DWT expects an even-length [batch, time, channel] tensor")
    pairs = values.reshape(values.shape[0], -1, 2, values.shape[2])
    return ((pairs[:, :, 0] + pairs[:, :, 1]) / math.sqrt(2),
            (pairs[:, :, 0] - pairs[:, :, 1]) / math.sqrt(2))


def haar_idwt(low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    if low.shape != high.shape or low.ndim != 3:
        raise ValueError("Low and high coefficients must have the same [batch, time, channel] shape")
    values = torch.stack(((low + high) / math.sqrt(2),
                          (low - high) / math.sqrt(2)), dim=2)
    return values.reshape(low.shape[0], low.shape[1] * 2, low.shape[2])


class WaveletAttention(nn.Module):
    """WTFTP temporal enhancement followed by scale-aligned convolution."""

    def __init__(self, seq_len: int, embedding: int):
        super().__init__()
        self.compress = nn.Sequential(nn.Linear(embedding, embedding // 2, bias=False),
                                      nn.ReLU(), nn.Linear(embedding // 2, 1, bias=False))
        self.excite = nn.Sequential(nn.Linear(seq_len, seq_len // 2, bias=False),
                                    nn.ReLU(), nn.Linear(seq_len // 2, seq_len, bias=False),
                                    nn.Sigmoid())
        # The original 9->10 forecast uses padding=1 to obtain five coefficients.
        # Same-window 96->96 reconstruction needs exactly 48 coefficients.
        self.align = nn.Sequential(nn.Conv1d(embedding, embedding, kernel_size=2,
                                             stride=2, padding=0, bias=False), nn.ReLU())

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        weights = self.excite(self.compress(encoded).transpose(1, 2)).transpose(1, 2)
        enhanced = encoded * (1 + weights)
        return self.align(enhanced.transpose(1, 2)).transpose(1, 2)


class WTFTPReconstruction(nn.Module):
    def __init__(self, config: WTFTPReconstructionConfig = WTFTPReconstructionConfig()):
        super().__init__()
        if (config.seq_len < 2 or config.seq_len % 2 or config.channels < 1 or
                config.embedding < 2 or config.embedding % 2 or config.encoder_layers < 1 or
                config.decoder_layers != 1 or not 0 <= config.dropout < 1 or
                config.wavelet != "haar" or config.levels != 1):
            raise ValueError("This WTFTP reconstruction adaptation supports one-level Haar, "
                             "even windows, and one-layer coefficient decoders")
        self.config = config
        dim = config.embedding
        self.embedding = nn.Sequential(nn.Linear(config.channels, dim // 2, bias=False), nn.ReLU(),
                                       nn.Linear(dim // 2, dim, bias=False), nn.ReLU())
        self.encoder = nn.LSTM(dim, dim, config.encoder_layers, batch_first=True,
                               dropout=config.dropout if config.encoder_layers > 1 else 0, bias=False)
        self.attentions = nn.ModuleList(WaveletAttention(config.seq_len, dim) for _ in range(2))
        self.decoders = nn.ModuleList(nn.LSTM(dim, dim, config.decoder_layers,
                                              batch_first=True, bias=False) for _ in range(2))
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(2))
        self.projections = nn.ModuleList(nn.Linear(dim, config.channels) for _ in range(2))

    def coefficients(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 3 or values.shape[1:] != (self.config.seq_len, self.config.channels):
            raise ValueError(f"Expected [batch, {self.config.seq_len}, {self.config.channels}]")
        encoded, (_, encoder_cell) = self.encoder(self.embedding(values))
        outputs = []
        for attention, decoder, norm, projection in zip(
                self.attentions, self.decoders, self.norms, self.projections):
            attended = attention(encoded)
            initial_hidden = torch.zeros(1, values.shape[0], self.config.embedding,
                                         device=values.device, dtype=values.dtype)
            initial_cell = encoder_cell[-1:].contiguous()
            decoded, _ = decoder(attended, (initial_hidden, initial_cell))
            outputs.append(projection(norm(decoded)))
        return outputs[0], outputs[1]

    def reconstruct(self, values: torch.Tensor) -> torch.Tensor:
        return haar_idwt(*self.coefficients(values))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        true_low, true_high = haar_dwt(values)
        low, high = self.coefficients(values)
        return F.mse_loss(low, true_low) + F.mse_loss(high, true_high)

    def score(self, values: torch.Tensor) -> torch.Tensor:
        return (self.reconstruct(values) - values).square()
