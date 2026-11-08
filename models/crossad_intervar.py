"""Experimental CrossAD variant with same-patch inter-variable attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from models.crossad import Attention, CrossAD, CrossADConfig, downsample, patchify


@dataclass(frozen=True)
class InterVarConfig(CrossADConfig):
    channels: int = 18


class CrossChannelMixer(nn.Module):
    def __init__(self, config: InterVarConfig):
        super().__init__()
        if config.channels < 2:
            raise ValueError("Cross-channel attention requires at least two channels")
        self.channels = config.channels
        self.embedding = nn.Parameter(torch.randn(1, config.channels, config.d_model) * 0.02)
        self.norm = nn.LayerNorm(config.d_model)
        self.attention = Attention(config.d_model, config.n_heads, config.dropout)
        self.gate = nn.Parameter(torch.tensor(0.1))
        self.register_buffer("exclude_self", torch.eye(config.channels, dtype=torch.bool)[None, None])

    def forward(self, encoded: torch.Tensor, batch: int, channels: int) -> torch.Tensor:
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")
        patches, dim = encoded.shape[1:]
        tokens = encoded.reshape(batch, channels, patches, dim).transpose(1, 2)
        tokens = tokens.reshape(batch * patches, channels, dim)
        context = self.norm(tokens + self.embedding)
        messages = self.attention(context, context, self.exclude_self)
        mixed = tokens + torch.tanh(self.gate) * messages
        return mixed.reshape(batch, patches, channels, dim).transpose(1, 2).reshape_as(encoded)


class CrossADInterVar(CrossAD):
    def __init__(self, config: InterVarConfig = InterVarConfig()):
        super().__init__(config)
        with torch.random.fork_rng(devices=[]):
            self.channel_mixer = CrossChannelMixer(config)

    def reconstruct(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, channels = x.shape
        if length != self.config.seq_len:
            raise ValueError(f"Expected {self.config.seq_len} points, got {length}")
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
        encoded = self.channel_mixer(encoded, batch, channels)
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
