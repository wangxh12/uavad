"""Independent implementation of the CrossAD architecture (Li et al., NeurIPS 2025)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CrossADConfig:
    seq_len: int = 192
    patch_len: int = 6
    ms_kernels: tuple[int, ...] = (32, 16, 8, 4, 2)
    d_model: int = 128
    n_heads: int = 4
    e_layers: int = 2
    d_layers: int = 2
    m_layers: int = 2
    n_query: int = 5
    query_len: int = 5
    topk: int = 10
    bank_size: int = 32
    decay: float = 0.95
    epsilon: float = 1e-5
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


def downsample(x: torch.Tensor, kernel: int) -> torch.Tensor:
    x = x.transpose(1, 2)
    x = F.pad(x, (0, kernel - 1), mode="replicate")
    return x.unfold(-1, kernel, kernel).mean(-1).transpose(1, 2)


def patchify(x: torch.Tensor, patch_len: int) -> torch.Tensor:
    x = F.pad(x.transpose(1, 2), (0, patch_len - 1), mode="replicate")
    return x.unfold(-1, patch_len, patch_len).flatten(0, 1)


def sinusoidal_positions(length: int, dim: int) -> torch.Tensor:
    positions = torch.arange(length).float().unsqueeze(-1)
    periods = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    encoded = torch.zeros(1, length, dim)
    encoded[..., 0::2] = torch.sin(positions * periods)
    encoded[..., 1::2] = torch.cos(positions * periods)
    return encoded


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, query: torch.Tensor, key: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch = query.shape[0]
        q = self.q(query).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(key).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(key).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            logits = logits.masked_fill(mask, -torch.inf)
        weights = self.attn_dropout(logits.softmax(-1))
        output = (weights @ v).transpose(1, 2).contiguous().view(batch, -1, self.heads * self.head_dim)
        return self.projection(output)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, kind: str):
        super().__init__()
        self.kind = kind
        self.attention = Attention(dim, heads, dropout)
        self.cross_attention = Attention(dim, heads, dropout) if kind == "decoder" else None
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim) if kind == "decoder" else None
        self.ff1 = nn.Conv1d(dim, 4 * dim, 1)
        self.ff2 = nn.Conv1d(4 * dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.kind == "extractor":
            x = self.norm1(x + self.dropout(self.attention(x, context)))
        else:
            x = self.norm1(x + self.dropout(self.attention(x, x, mask)))
            if self.kind == "decoder":
                x = self.norm2(x + self.dropout(self.cross_attention(x, context)))
            else:
                # Match the reference encoder's second application of norm1.
                x = self.norm1(x)
        y = self.dropout(F.gelu(self.ff1(x.transpose(1, 2))))
        y = self.dropout(self.ff2(y).transpose(1, 2))
        norm = self.norm3 if self.kind == "decoder" else self.norm2
        return norm(x + y)


class ContextBank(nn.Module):
    def __init__(self, config: CrossADConfig, encoder_length: int):
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.seq_len, config.n_query)
        self.queries = nn.Parameter(torch.randn(config.n_query, config.query_len, config.d_model))
        self.extractor = nn.ModuleList(
            TransformerBlock(config.d_model, config.n_heads, config.dropout, "extractor")
            for _ in range(config.m_layers)
        )
        self.register_buffer("context", torch.randn(config.bank_size, config.query_len, config.d_model))
        self.register_buffer("ema_count", torch.ones(config.bank_size))
        self.register_buffer("ema_sum", torch.zeros_like(self.context))

    def forward(self, series: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return self.context.reshape(-1, self.config.d_model).unsqueeze(0).expand(len(series), -1, -1)
        frequencies = torch.fft.rfft(series, dim=1)
        indices = frequencies.abs().topk(self.config.topk, dim=1).indices
        retained = torch.zeros_like(frequencies).scatter(1, indices, frequencies.gather(1, indices))
        routed = torch.fft.irfft(retained, n=self.config.seq_len, dim=1)
        choice = F.gumbel_softmax(self.router(routed.flatten(1)), tau=1, hard=True)
        query = torch.einsum("bn,nqd->bqd", choice, self.queries)
        for block in self.extractor:
            query = block(query, encoded)

        flat_query = query.flatten(1)
        flat_bank = self.context.flatten(1)
        distances = (flat_query.square().sum(1, keepdim=True)
                     + flat_bank.square().sum(1) - 2 * flat_query @ flat_bank.T)
        assignment = F.one_hot(distances.argmin(1), self.config.bank_size).to(query.dtype)
        assigned_query = torch.einsum("bn,bqd->nqd", assignment, query)
        with torch.no_grad():
            decay = self.config.decay
            self.ema_count.mul_(decay).add_((1 - decay) * assignment.sum(0))
            total = self.ema_count.sum()
            dim = flat_bank.shape[1]
            self.ema_count.copy_((self.ema_count + self.config.epsilon)
                                 / (total + dim * self.config.epsilon) * total)
            self.ema_sum.mul_(decay).add_((1 - decay) * assigned_query)
            self.context.copy_(self.ema_sum / self.ema_count[:, None, None])
        # Straight-through query content, with the EMA bank used in the forward pass.
        content = assigned_query + self.context.detach() - assigned_query.detach()
        return content.reshape(-1, self.config.d_model).unsqueeze(0).expand(len(series), -1, -1)


class CrossAD(nn.Module):
    def __init__(self, config: CrossADConfig = CrossADConfig()):
        super().__init__()
        self.config = config
        self.time_lengths = [math.ceil(config.seq_len / k) for k in config.ms_kernels] + [config.seq_len]
        self.patch_lengths = [math.ceil(n / config.patch_len) for n in self.time_lengths]
        self.patch_embedding = nn.Linear(config.patch_len, config.d_model, bias=False)
        self.register_buffer("positions", sinusoidal_positions(self.patch_lengths[-1], config.d_model))

        encoder_scales = torch.cat([torch.full((n,), i) for i, n in enumerate(self.patch_lengths[:-1])])
        decoder_scales = torch.cat([torch.full((n,), i) for i, n in enumerate(self.patch_lengths[1:])])
        self.register_buffer("encoder_mask", (encoder_scales[:, None] != encoder_scales[None, :])[None, None])
        self.register_buffer("decoder_mask", (decoder_scales[:, None] < decoder_scales[None, :])[None, None])

        self.encoder = nn.ModuleList(
            TransformerBlock(config.d_model, config.n_heads, config.dropout, "encoder")
            for _ in range(config.e_layers)
        )
        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.bank = ContextBank(config, sum(self.patch_lengths[:-1]))
        self.decoder = nn.ModuleList(
            TransformerBlock(config.d_model, config.n_heads, config.dropout, "decoder")
            for _ in range(config.d_layers)
        )
        self.decoder_norm = nn.LayerNorm(config.d_model)
        self.projection = nn.Linear(config.d_model, config.patch_len)

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
        truth, reconstruction = self.reconstruct(x)
        return F.mse_loss(reconstruction, truth)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        truth, reconstruction = self.reconstruct(x)
        errors = (reconstruction - truth).square().split(self.time_lengths[1:], dim=1)
        score = errors[-1].clone()
        for coarse in errors[:-1]:
            score += F.interpolate(coarse.transpose(1, 2), size=self.config.seq_len,
                                   mode="linear").transpose(1, 2)
        return score
