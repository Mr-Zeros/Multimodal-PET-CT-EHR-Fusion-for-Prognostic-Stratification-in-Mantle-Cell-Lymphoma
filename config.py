from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PreprocessingConfig:
    target_shape: tuple[int, int, int] = (24, 96, 96)
    registration_shape: tuple[int, int, int] = (16, 64, 64)
    pet_clip_range: tuple[float, float] = (0.0, 25.0)
    ct_clip_range: tuple[float, float] = (-1000.0, 1000.0)
    default_voxel_spacing_mm: tuple[float, float, float] = (4.0, 4.0, 4.0)
    voi_threshold_fraction: float = 0.41
    registration_bins: int = 24
    registration_steps: int = 12
    registration_lr: float = 0.03
    registration_regularization: float = 5e-3
    body_mask_percentile: float = 35.0
    epsilon: float = 1e-6


@dataclass(slots=True)
class MoEConfig:
    embed_dim: int = 192
    transformer_heads: int = 4
    transformer_layers: int = 2
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    radiomics_features: int = 18
    text_vocab_size: int = 4096
    max_image_slices: int = 32
    max_sentences: int = 48
    max_timestamp_buckets: int = 32
    group_order: tuple[str, ...] = (
        "MedCLIP-PET",
        "MedCLIP-CT",
        "MedSAM-PET",
        "MedSAM-CT",
        "Radiomics-PET",
        "Radiomics-CT",
        "MedBERT-Text",
    )


@dataclass(slots=True)
class ModelConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
