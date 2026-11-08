"""Multiscale lagged channel fusion for CrossAD."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from models.crossad import CrossAD, CrossADConfig, downsample, patchify


@dataclass(frozen=True)
class RelationConfig(CrossADConfig):
    channels: int = 18
    neighbors: int = 4
    lag: int = 1
    mask_weight: float = 0.2


class LaggedVariableMixer(nn.Module):
    def __init__(self, config: RelationConfig):
        super().__init__()
        if not 1 <= config.neighbors < config.channels:
            raise ValueError("neighbors must be between 1 and channels - 1")
        if config.lag < 0:
            raise ValueError("lag must be nonnegative")
        self.channels = config.channels
        self.lag = config.lag
        self.heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.neighbors = config.neighbors
        scales = len(config.ms_kernels)
        self.embedding = nn.Parameter(torch.randn(config.channels, config.d_model) * 0.02)
        self.graph = nn.Parameter(torch.randn(scales, config.channels, config.channels) * 0.01)
        self.gate = nn.Parameter(torch.full((scales, config.channels, 1), 0.1))
        self.query = nn.Linear(config.d_model, config.d_model)
        self.key = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        self.output = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.Dropout(config.dropout))
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, encoded: torch.Tensor, batch: int, lengths: list[int]) -> torch.Tensor:
        channels = self.channels
        if encoded.shape[0] != batch * channels or encoded.shape[1] != sum(lengths):
            raise ValueError("Expected channel-major encoded patches at every configured scale")
        tokens = encoded.reshape(batch, channels, -1, encoded.shape[-1])
        chunks = tokens.split(lengths, dim=2)
        result = [self._mix(chunk.transpose(1, 2), scale).transpose(1, 2)
                  for scale, chunk in enumerate(chunks)]
        return torch.cat(result, dim=2).reshape_as(encoded)

    def _mix(self, tokens: torch.Tensor, scale: int) -> torch.Tensor:
        batch, patches, channels, dim = tokens.shape
        source = self.norm(tokens + self.embedding)
        offsets = range(-self.lag, self.lag + 1)
        positions = torch.arange(patches, device=tokens.device)
        shifted = torch.stack([source[:, (positions + offset).clamp(0, patches - 1)]
                               for offset in offsets], dim=3)
        lag_count = len(offsets)
        keys = shifted.reshape(batch, patches, channels * lag_count, dim)

        graph = self.graph[scale]
        selectable = graph.masked_fill(torch.eye(channels, dtype=torch.bool, device=tokens.device),
                                       -torch.inf)
        chosen = selectable.topk(self.neighbors, dim=-1).indices
        edges = torch.zeros_like(graph, dtype=torch.bool).scatter_(1, chosen, True)
        valid = torch.stack([((positions + offset >= 0) & (positions + offset < patches))
                             for offset in offsets], dim=-1)
        allowed = (edges[None, :, :, None] & valid[:, None, None, :]).reshape(
            patches, channels, channels * lag_count)

        q = self.query(source).reshape(batch, patches, channels, self.heads, self.head_dim).transpose(2, 3)
        k = self.key(keys).reshape(batch, patches, -1, self.heads, self.head_dim).transpose(2, 3)
        v = self.value(keys).reshape(batch, patches, -1, self.heads, self.head_dim).transpose(2, 3)
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        logits = logits + graph.repeat_interleave(lag_count, dim=1)[None, None, None]
        logits = logits.masked_fill(~allowed[None, :, None], -torch.inf)
        weights = self.dropout(logits.softmax(dim=-1))
        message = (weights @ v).transpose(2, 3).reshape(batch, patches, channels, dim)
        return tokens + torch.tanh(self.gate[scale])[None, None] * self.output(message)


class CrossADRelation(CrossAD):
    def __init__(self, config: RelationConfig = RelationConfig()):
        super().__init__(config)
        if config.mask_weight < 0:
            raise ValueError("mask_weight must be nonnegative")
        self.mixer = LaggedVariableMixer(config)
        self.mask_embedding = nn.Parameter(torch.randn(config.d_model) * 0.02)

    def reconstruct(self, x: torch.Tensor, masked_targets: torch.Tensor | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, channels = x.shape
        if length != self.config.seq_len or channels != self.config.channels:
            raise ValueError(f"Expected [batch, {self.config.seq_len}, {self.config.channels}]")
        series = x.transpose(1, 2).reshape(batch * channels, length, 1)
        scales = [downsample(series, k) for k in self.config.ms_kernels]
        truth = torch.cat(scales[1:] + [series], dim=1)
        truth = truth.reshape(batch, channels, -1).transpose(1, 2)

        embedded = [self.patch_embedding(patchify(item, self.config.patch_len)) for item in scales]
        position_scales = [downsample(self.positions, k) for k in self.config.ms_kernels]
        encoded = torch.cat(embedded, dim=1) + torch.cat(position_scales, dim=1)
        for block in self.encoder:
            encoded = block(encoded, mask=self.encoder_mask)
        encoded = self.encoder_norm(encoded)
        if masked_targets is not None:
            indicator = F.one_hot(masked_targets, channels).to(encoded.dtype)
            encoded = encoded + (indicator[:, :, None, None] * self.mask_embedding).reshape(
                batch * channels, 1, -1)
        encoded = self.mixer(encoded, batch, self.patch_lengths[:-1])
        context = self.bank(series, encoded)

        segments = encoded.split(self.patch_lengths[:-1], dim=1)
        enlarged = [F.interpolate(segment.transpose(1, 2), size=self.patch_lengths[i + 1],
                                  mode="nearest").transpose(1, 2) for i, segment in enumerate(segments)]
        decoded = torch.cat(enlarged, dim=1)
        for block in self.decoder:
            decoded = block(decoded, context, self.decoder_mask)
        decoded = self.projection(self.decoder_norm(decoded)).flatten(1, 2).unsqueeze(-1)
        decoded = decoded.reshape(batch, channels, -1).transpose(1, 2)
        chunks = decoded.split([n * self.config.patch_len for n in self.patch_lengths[1:]], dim=1)
        reconstruction = torch.cat([chunk[:, :actual] for chunk, actual in
                                    zip(chunks, self.time_lengths[1:])], dim=1)
        return truth, reconstruction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.config.mask_weight == 0:
            return super().forward(x)
        batch = x.shape[0]
        targets = torch.randint(self.config.channels, (batch,), device=x.device)
        masked = x.clone()
        masked[torch.arange(batch, device=x.device), :, targets] = 0

        # The auxiliary pass must not update the normal context bank with masked inputs.
        self.bank.eval()
        try:
            _, conditional = self.reconstruct(masked, targets)
        finally:
            self.bank.train()
        truth, reconstruction = self.reconstruct(x)
        selected = targets[:, None, None].expand(-1, truth.shape[1], 1)
        auxiliary = F.mse_loss(conditional.gather(2, selected), truth.gather(2, selected))
        return F.mse_loss(reconstruction, truth) + self.config.mask_weight * auxiliary
