from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class StructuredSentence:
    text: str
    section: str = "UNKNOWN"
    negated: bool = False
    temporality: str = "present"
    timestamp: str | None = None


@dataclass(slots=True)
class CaseSample:
    case_id: str
    pet_volume: torch.Tensor
    ct_volume: torch.Tensor
    pet_voi: torch.Tensor | None = None
    ct_voi: torch.Tensor | None = None
    sentences: list[StructuredSentence] = field(default_factory=list)


@dataclass(slots=True)
class ExpertBatch:
    group_tokens: dict[str, torch.Tensor]
    group_padding_mask: dict[str, torch.Tensor]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FusionOutput:
    patient_embedding: torch.Tensor
    group_vectors: dict[str, torch.Tensor]
    refined_group_vectors: dict[str, torch.Tensor]
    gating_weights: torch.Tensor
    intra_group_attention: dict[str, list[torch.Tensor]]
    pooling_weights: dict[str, torch.Tensor]
    cross_group_attention: list[torch.Tensor]


@dataclass(slots=True)
class EndpointOutput:
    risk: torch.Tensor
    r_signature: torch.Tensor
    fusion: FusionOutput


@dataclass(slots=True)
class PETMetrics:
    suvmax: torch.Tensor
    suvmean: torch.Tensor
    tmtv: torch.Tensor
    tlg: torch.Tensor


@dataclass(slots=True)
class ClinicalCovariates:
    tlg: torch.Tensor
    wbc_elevated: torch.Tensor | None = None
    ki67_high: torch.Tensor | None = None
    beta2_microglobulin_elevated: torch.Tensor | None = None
