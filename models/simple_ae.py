"""Small multivariate reconstruction baselines for ALFA."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ReconstructionBaseline(nn.Module):
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.reconstruct(x), x)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return (self.reconstruct(x) - x).square()


class LSTMAE(ReconstructionBaseline):
    def __init__(self, features: int = 18, hidden: int = 64, latent: int = 32):
        super().__init__()
        self.encoder = nn.LSTM(features, hidden, batch_first=True)
        self.compress = nn.Linear(hidden, latent)
        self.decoder = nn.LSTM(latent, hidden, batch_first=True)
        self.output = nn.Linear(hidden, features)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        _, (state, _) = self.encoder(x)
        code = torch.tanh(self.compress(state[-1]))
        decoded, _ = self.decoder(code[:, None].expand(-1, x.size(1), -1))
        return self.output(decoded)


class MLPAE(ReconstructionBaseline):
    def __init__(self, length: int = 192, features: int = 18, hidden: int = 256,
                 latent: int = 64):
        super().__init__()
        self.length = length
        self.features = features
        flat = length * features
        self.network = nn.Sequential(nn.Linear(flat, hidden), nn.GELU(),
                                     nn.Linear(hidden, latent), nn.GELU(),
                                     nn.Linear(latent, hidden), nn.GELU(),
                                     nn.Linear(hidden, flat))

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1:] != (self.length, self.features):
            raise ValueError(f"Expected [batch, {self.length}, {self.features}]")
        return self.network(x.flatten(1)).reshape_as(x)


class PCABaseline(ReconstructionBaseline):
    def __init__(self, features: int = 18, components: int = 8):
        super().__init__()
        self.register_buffer("mean", torch.zeros(features))
        self.register_buffer("basis", torch.zeros(components, features))

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - self.mean
        return (centered @ self.basis.T) @ self.basis + self.mean
