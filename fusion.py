from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import MoEConfig
from .data import ExpertBatch, FusionOutput


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.transformer_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.transformer_layers)
            ]
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        attention_maps: list[torch.Tensor] = []
        for layer in self.layers:
            x, attn = layer(x, key_padding_mask=key_padding_mask)
            attention_maps.append(attn)
        return x, attention_maps


class GlobalQueryAttentionPooling(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(embed_dim))

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.einsum("bte,e->bt", tokens, self.query) / math.sqrt(tokens.size(-1))
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bt,bte->be", weights, tokens)
        return pooled, weights


class HierarchicalMoEFusion(nn.Module):
    def __init__(self, config: MoEConfig | None = None) -> None:
        super().__init__()
        self.config = config or MoEConfig()
        self.intra_group_encoders = nn.ModuleDict(
            {group_name: TransformerEncoder(self.config) for group_name in self.config.group_order}
        )
        self.intra_group_poolers = nn.ModuleDict(
            {group_name: GlobalQueryAttentionPooling(self.config.embed_dim) for group_name in self.config.group_order}
        )
        self.cross_group_encoder = TransformerEncoder(self.config)
        self.gating_network = nn.Sequential(
            nn.LayerNorm(self.config.embed_dim * len(self.config.group_order)),
            nn.Linear(self.config.embed_dim * len(self.config.group_order), self.config.embed_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.embed_dim, len(self.config.group_order)),
        )

    def forward(self, expert_batch: ExpertBatch) -> FusionOutput:
        group_vectors: dict[str, torch.Tensor] = {}
        refined_group_vectors: dict[str, torch.Tensor] = {}
        intra_group_attention: dict[str, list[torch.Tensor]] = {}
        pooling_weights: dict[str, torch.Tensor] = {}

        ordered_vectors = []
        for group_name in self.config.group_order:
            group_tokens = expert_batch.group_tokens[group_name]
            group_padding_mask = expert_batch.group_padding_mask[group_name]
            contextualized, attn_maps = self.intra_group_encoders[group_name](group_tokens, group_padding_mask)
            pooled, pool_weights = self.intra_group_poolers[group_name](contextualized, group_padding_mask)
            group_vectors[group_name] = pooled
            intra_group_attention[group_name] = attn_maps
            pooling_weights[group_name] = pool_weights
            ordered_vectors.append(pooled)

        stacked_vectors = torch.stack(ordered_vectors, dim=1)
        refined_stack, cross_group_attention = self.cross_group_encoder(stacked_vectors)
        gate_input = refined_stack.flatten(start_dim=1)
        gating_weights = torch.softmax(self.gating_network(gate_input), dim=-1)
        patient_embedding = torch.sum(refined_stack * gating_weights.unsqueeze(-1), dim=1)

        for group_index, group_name in enumerate(self.config.group_order):
            refined_group_vectors[group_name] = refined_stack[:, group_index]

        return FusionOutput(
            patient_embedding=patient_embedding,
            group_vectors=group_vectors,
            refined_group_vectors=refined_group_vectors,
            gating_weights=gating_weights,
            intra_group_attention=intra_group_attention,
            pooling_weights=pooling_weights,
            cross_group_attention=cross_group_attention,
        )
