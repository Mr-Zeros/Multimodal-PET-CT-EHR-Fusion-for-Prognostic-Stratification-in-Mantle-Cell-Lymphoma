from __future__ import annotations

from collections import defaultdict

import torch

from .data import ExpertBatch, FusionOutput
from .survival import EndpointTaskModel


def attention_rollout(attention_maps: list[torch.Tensor]) -> torch.Tensor:
    if not attention_maps:
        raise ValueError("attention_maps must not be empty")
    batch_size, _, token_count, _ = attention_maps[0].shape
    device = attention_maps[0].device
    rollout = torch.eye(token_count, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    for attention in attention_maps:
        attn = attention.mean(dim=1)
        identity = torch.eye(token_count, device=device).unsqueeze(0)
        attn = attn + identity
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        rollout = torch.bmm(attn, rollout)
    return rollout


def slice_token_importance(fusion_output: FusionOutput, group_name: str) -> torch.Tensor:
    rollout = attention_rollout(fusion_output.intra_group_attention[group_name])
    pooling = fusion_output.pooling_weights[group_name]
    importance = torch.bmm(pooling.unsqueeze(1), rollout).squeeze(1)
    importance = importance / importance.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return importance


def volume_heatmap(volume: torch.Tensor, token_importance: torch.Tensor) -> torch.Tensor:
    weights = token_importance.unsqueeze(-1).unsqueeze(-1)
    heatmap = volume * weights
    max_abs = heatmap.abs().flatten(start_dim=1).amax(dim=-1, keepdim=True).view(-1, 1, 1, 1)
    return heatmap / max_abs.clamp_min(1e-6)


def modality_level_contributions(gating_weights: torch.Tensor, group_order: tuple[str, ...]) -> dict[str, torch.Tensor]:
    modality_scores: dict[str, list[torch.Tensor]] = defaultdict(list)
    for group_index, group_name in enumerate(group_order):
        if "Text" in group_name:
            modality_scores["EHR"].append(gating_weights[:, group_index])
        elif "PET" in group_name:
            modality_scores["PET"].append(gating_weights[:, group_index])
        else:
            modality_scores["CT"].append(gating_weights[:, group_index])
    return {name: torch.stack(scores, dim=0).sum(dim=0) for name, scores in modality_scores.items()}


def ablate_groups(
    endpoint_model: EndpointTaskModel,
    expert_batch: ExpertBatch,
    drop_groups: set[str],
):
    ablated_tokens = {}
    for group_name, tokens in expert_batch.group_tokens.items():
        ablated_tokens[group_name] = torch.zeros_like(tokens) if group_name in drop_groups else tokens
    ablated_batch = ExpertBatch(
        group_tokens=ablated_tokens,
        group_padding_mask=expert_batch.group_padding_mask,
        metadata=expert_batch.metadata,
    )
    return endpoint_model(ablated_batch)


def risk_distribution_by_subtype(risk_score: torch.Tensor, subtypes: list[str]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for score, subtype in zip(risk_score.tolist(), subtypes, strict=True):
        groups[subtype].append(score)
    summary = {}
    for subtype, scores in groups.items():
        tensor = torch.tensor(scores, dtype=torch.float32)
        summary[subtype] = {
            "n": float(tensor.numel()),
            "mean": float(tensor.mean()),
            "std": float(tensor.std(unbiased=False)) if tensor.numel() > 1 else 0.0,
            "median": float(torch.median(tensor)),
        }
    return summary
