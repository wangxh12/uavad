"""Two cross-variable CrossAD variants with the original scoring interface."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from models.crossad import Attention, CrossAD, CrossADConfig, downsample, patchify


@dataclass(frozen=True)
class DualConfig(CrossADConfig):
    channels: int = 18
    conditional_weight: float = 0.2


@dataclass(frozen=True)
class CrossScaleConfig(CrossADConfig):
    channels: int = 18
    lag: int = 1


class _MultivariateCrossAD(CrossAD):
    def __init__(self, config: DualConfig | CrossScaleConfig):
        super().__init__(config)
        if config.channels < 2:
            raise ValueError("Cross-variable models require at least two channels")

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, channels = x.shape
        if length != self.config.seq_len or channels != self.config.channels:
            raise ValueError(f"Expected [batch, {self.config.seq_len}, {self.config.channels}]")
        series = x.transpose(1, 2).reshape(batch * channels, length, 1)
        scales = [downsample(series, k) for k in self.config.ms_kernels]
        truth = torch.cat(scales[1:] + [series], dim=1)
        truth = truth.reshape(batch, channels, -1).transpose(1, 2)
        embedded = [self.patch_embedding(patchify(item, self.config.patch_len)) for item in scales]
        positions = [downsample(self.positions, k) for k in self.config.ms_kernels]
        encoded = torch.cat(embedded, dim=1) + torch.cat(positions, dim=1)
        for block in self.encoder:
            encoded = block(encoded, mask=self.encoder_mask)
        return truth, series, self.encoder_norm(encoded)

    def _upscale(self, encoded: torch.Tensor) -> torch.Tensor:
        segments = encoded.split(self.patch_lengths[:-1], dim=1)
        enlarged = [F.interpolate(segment.transpose(1, 2), size=self.patch_lengths[i + 1],
                                  mode="nearest").transpose(1, 2)
                    for i, segment in enumerate(segments)]
        return torch.cat(enlarged, dim=1)

    def _project(self, decoded: torch.Tensor, head: nn.Module, batch: int) -> torch.Tensor:
        channels = self.config.channels
        projected = head(decoded).flatten(1, 2).reshape(batch, channels, -1).transpose(1, 2)
        chunks = projected.split([n * self.config.patch_len for n in self.patch_lengths[1:]], dim=1)
        return torch.cat([chunk[:, :actual] for chunk, actual in
                          zip(chunks, self.time_lengths[1:])], dim=1)

    def _decode(self, encoded: torch.Tensor, series: torch.Tensor,
                seeds: torch.Tensor, batch: int) -> torch.Tensor:
        context = self.bank(series, encoded)
        for block in self.decoder:
            seeds = block(seeds, context, self.decoder_mask)
        return self._project(self.decoder_norm(seeds), self.projection, batch)


class OtherChannelContext(nn.Module):
    """A target-independent query attends only to other channels at the same patch."""

    def __init__(self, config: DualConfig):
        super().__init__()
        self.channels = config.channels
        self.query = nn.Parameter(torch.randn(config.channels, config.d_model) * 0.02)
        self.channel_embedding = nn.Parameter(torch.randn(config.channels, config.d_model) * 0.02)
        self.norm = nn.LayerNorm(config.d_model)
        self.attention = Attention(config.d_model, config.n_heads, config.dropout)
        self.register_buffer("exclude_self", torch.eye(config.channels, dtype=torch.bool)[None, None])

    def forward(self, encoded: torch.Tensor, batch: int) -> torch.Tensor:
        channels, dim = self.channels, encoded.shape[-1]
        patches = encoded.shape[1]
        tokens = encoded.reshape(batch, channels, patches, dim).transpose(1, 2)
        tokens = tokens.reshape(batch * patches, channels, dim)
        queries = self.query[None].expand(batch * patches, -1, -1)
        sources = self.norm(tokens + self.channel_embedding[None])
        conditional = self.attention(queries, sources, self.exclude_self)
        return conditional.reshape(batch, patches, channels, dim).transpose(1, 2).reshape_as(encoded)


class CrossADDual(_MultivariateCrossAD):
    """Self reconstruction plus target-blind cross-channel conditional reconstruction."""

    def __init__(self, config: DualConfig = DualConfig()):
        super().__init__(config)
        if config.conditional_weight < 0:
            raise ValueError("conditional_weight must be nonnegative")
        self.other_channels = OtherChannelContext(config)
        self.gate = nn.Parameter(torch.full((len(config.ms_kernels), config.channels), 0.1))
        self.conditional_head = nn.Sequential(nn.LayerNorm(config.d_model),
                                              nn.Linear(config.d_model, config.patch_len))

    def _outputs(self, x: torch.Tensor, conditional_output: bool
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch = x.shape[0]
        truth, series, encoded = self._encode(x)
        conditional = self.other_channels(encoded, batch)
        own_seeds, other_seeds = self._upscale(encoded), self._upscale(conditional)
        own_parts = own_seeds.split(self.patch_lengths[1:], dim=1)
        other_parts = other_seeds.split(self.patch_lengths[1:], dim=1)
        fused = []
        for scale, (own, other) in enumerate(zip(own_parts, other_parts)):
            length, dim = own.shape[1:]
            gate = torch.tanh(self.gate[scale])[None, :, None, None]
            mixed = own.reshape(batch, self.config.channels, length, dim) + gate * other.reshape(
                batch, self.config.channels, length, dim)
            fused.append(mixed.reshape_as(own))
        reconstruction = self._decode(encoded, series, torch.cat(fused, dim=1), batch)
        auxiliary = (self._project(other_seeds, self.conditional_head, batch)
                     if conditional_output else None)
        return truth, reconstruction, auxiliary

    def reconstruct(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        truth, reconstruction, _ = self._outputs(x, False)
        return truth, reconstruction

    def conditional_reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        _, _, encoded = self._encode(x)
        conditional = self.other_channels(encoded, x.shape[0])
        return self._project(self._upscale(conditional), self.conditional_head, x.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        truth, reconstruction, auxiliary = self._outputs(
            x, self.training and self.config.conditional_weight > 0)
        loss = F.mse_loss(reconstruction, truth)
        if auxiliary is not None:
            loss = loss + self.config.conditional_weight * F.mse_loss(auxiliary, truth)
        return loss


class CrossScaleContext(nn.Module):
    """Query a target's coarse patch against other channels' neighboring coarse patches."""

    def __init__(self, config: CrossScaleConfig):
        super().__init__()
        if config.lag < 0:
            raise ValueError("lag must be nonnegative")
        self.channels = config.channels
        self.lag = config.lag
        self.heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.embedding = nn.Parameter(torch.randn(config.channels, config.d_model) * 0.02)
        self.pair_bias = nn.Parameter(torch.zeros(len(config.ms_kernels), config.channels, config.channels))
        self.gate = nn.Parameter(torch.full((len(config.ms_kernels), config.channels), 0.1))
        self.norm = nn.LayerNorm(config.d_model)
        self.query = nn.Linear(config.d_model, config.d_model)
        self.key = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        self.output = nn.Sequential(nn.Linear(config.d_model, config.d_model),
                                    nn.Dropout(config.dropout))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, coarse: torch.Tensor, fine_length: int, scale: int) -> torch.Tensor:
        batch, length, channels, dim = coarse.shape
        parent = torch.div(torch.arange(fine_length, device=coarse.device) * length,
                           fine_length, rounding_mode="floor")
        offsets = torch.arange(-self.lag, self.lag + 1, device=coarse.device)
        neighbor_positions = parent[:, None] + offsets[None]
        valid = (neighbor_positions >= 0) & (neighbor_positions < length)
        source = self.norm(coarse + self.embedding[None, None])
        neighbors = torch.stack([source[:, neighbor_positions[:, index].clamp(0, length - 1)]
                                 for index in range(len(offsets))], dim=3)
        keys = neighbors.reshape(batch, fine_length, channels * len(offsets), dim)
        query = source[:, parent]
        q = self.query(query).reshape(batch, fine_length, channels, self.heads,
                                      self.head_dim).transpose(2, 3)
        k = self.key(keys).reshape(batch, fine_length, -1, self.heads,
                                   self.head_dim).transpose(2, 3)
        v = self.value(keys).reshape(batch, fine_length, -1, self.heads,
                                     self.head_dim).transpose(2, 3)
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        bias = self.pair_bias[scale].repeat_interleave(len(offsets), dim=-1)
        logits = logits + bias[None, None, None]
        not_self = ~torch.eye(channels, dtype=torch.bool, device=coarse.device)
        allowed = (not_self[:, :, None] & valid[:, None, None, :]).reshape(
            fine_length, channels, channels * len(offsets))
        weights = self.dropout(logits.masked_fill(~allowed[None, :, None], -torch.inf).softmax(-1))
        message = (weights @ v).transpose(2, 3).reshape(batch, fine_length, channels, dim)
        return self.output(message)


class CrossADCrossScale(_MultivariateCrossAD):
    """Use cross-channel coarse context to generate each finer-scale decoder seed."""

    def __init__(self, config: CrossScaleConfig = CrossScaleConfig()):
        super().__init__(config)
        self.cross_scale = CrossScaleContext(config)

    def reconstruct(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        truth, series, encoded = self._encode(x)
        segments = encoded.split(self.patch_lengths[:-1], dim=1)
        seeds = []
        for scale, segment in enumerate(segments):
            fine_length = self.patch_lengths[scale + 1]
            own = F.interpolate(segment.transpose(1, 2), size=fine_length,
                                mode="nearest").transpose(1, 2)
            coarse = segment.reshape(batch, self.config.channels, segment.shape[1], -1).transpose(1, 2)
            other = self.cross_scale(coarse, fine_length, scale).transpose(1, 2)
            gate = torch.tanh(self.cross_scale.gate[scale])[None, :, None, None]
            mixed = own.reshape(batch, self.config.channels, fine_length, -1) + gate * other
            seeds.append(mixed.reshape_as(own))
        return truth, self._decode(encoded, series, torch.cat(seeds, dim=1), batch)
