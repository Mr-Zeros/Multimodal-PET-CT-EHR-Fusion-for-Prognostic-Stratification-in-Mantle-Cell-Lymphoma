from __future__ import annotations

import hashlib
from datetime import datetime
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MoEConfig
from .data import ExpertBatch, StructuredSentence
from .preprocessing import PETCTPreprocessor


def _stable_hash(text: str, vocab_size: int) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % vocab_size


class SliceCNN(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        hidden = max(embed_dim // 2, 32)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden // 2, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm2d(hidden // 2),
            nn.GELU(),
            nn.Conv2d(hidden // 2, hidden, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(embed_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return features.flatten(1)


class MedCLIPVisionExpert(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.backbone = SliceCNN(in_channels=1, embed_dim=embed_dim)
        self.projection = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))

    def forward(self, volume: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, depth, height, width = volume.shape
        slices = volume.reshape(batch * depth, 1, height, width)
        tokens = self.projection(self.backbone(slices)).reshape(batch, depth, -1)
        padding_mask = torch.zeros(batch, depth, device=volume.device, dtype=torch.bool)
        return tokens, padding_mask


class MedSAMVisionExpert(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.backbone = SliceCNN(in_channels=3, embed_dim=embed_dim)
        self.projection = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))

    def _derive_mask(self, volume: torch.Tensor) -> torch.Tensor:
        flat = volume.flatten(start_dim=2)
        threshold = torch.quantile(flat, 0.85, dim=-1, keepdim=True).unsqueeze(-1)
        return volume >= threshold

    def _edge_map(self, mask: torch.Tensor) -> torch.Tensor:
        mask_5d = mask.float().unsqueeze(1)
        kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=mask.device)
        kernel_y = kernel_x.t()
        kernel_x = kernel_x.view(1, 1, 3, 3)
        kernel_y = kernel_y.view(1, 1, 3, 3)
        edges = []
        for slice_mask in mask_5d.unbind(dim=2):
            grad_x = F.conv2d(slice_mask, kernel_x, padding=1)
            grad_y = F.conv2d(slice_mask, kernel_y, padding=1)
            edges.append((grad_x.square() + grad_y.square()).sqrt())
        return torch.stack(edges, dim=2)[:, 0]

    def forward(self, volume: torch.Tensor, voi_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if voi_mask is None:
            voi_mask = self._derive_mask(volume)
        edge_map = self._edge_map(voi_mask)
        batch, depth, height, width = volume.shape
        stacked = torch.stack([volume, voi_mask.float(), edge_map], dim=2)
        slices = stacked.reshape(batch * depth, 3, height, width)
        tokens = self.projection(self.backbone(slices)).reshape(batch, depth, -1)
        padding_mask = torch.zeros(batch, depth, device=volume.device, dtype=torch.bool)
        return tokens, padding_mask


class RadiomicsExpert(nn.Module):
    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Sequential(
            nn.Linear(config.radiomics_features, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
        )

    def _derive_mask(self, slice_tensor: torch.Tensor) -> torch.Tensor:
        threshold = torch.quantile(slice_tensor.flatten(), 0.80)
        return slice_tensor >= threshold

    def _entropy(self, values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return values.new_tensor(0.0)
        min_val = values.min()
        max_val = values.max()
        if torch.isclose(min_val, max_val):
            return values.new_tensor(0.0)
        hist = torch.histc(values, bins=16, min=float(min_val), max=float(max_val))
        probs = hist / hist.sum().clamp_min(1.0)
        probs = probs.clamp_min(1e-6)
        return -(probs * probs.log()).sum()

    def _single_slice_features(self, slice_tensor: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        mask = self._derive_mask(slice_tensor) if mask is None else mask.bool()
        values = slice_tensor[mask]
        if values.numel() == 0:
            values = slice_tensor.flatten()
            mask = torch.ones_like(slice_tensor, dtype=torch.bool)
        mean = values.mean()
        std = values.std(unbiased=False)
        centered = values - mean
        std_eps = std.clamp_min(1e-6)
        skewness = (centered.pow(3).mean() / std_eps.pow(3)).clamp(-50.0, 50.0)
        kurtosis = (centered.pow(4).mean() / std_eps.pow(4)).clamp(0.0, 100.0)
        gradients = torch.gradient(slice_tensor.float(), dim=(0, 1))
        grad_mag = torch.sqrt(gradients[0].square() + gradients[1].square())
        y_idx, x_idx = torch.where(mask)
        center_y = y_idx.float().mean() / max(slice_tensor.shape[0] - 1, 1) if y_idx.numel() > 0 else values.new_tensor(0.5)
        center_x = x_idx.float().mean() / max(slice_tensor.shape[1] - 1, 1) if x_idx.numel() > 0 else values.new_tensor(0.5)
        quantiles = torch.quantile(values, torch.tensor([0.25, 0.50, 0.75], device=values.device))
        features = torch.stack(
            [
                mean,
                std,
                values.min(),
                values.max(),
                quantiles[0],
                quantiles[1],
                quantiles[2],
                values.square().mean(),
                self._entropy(values),
                skewness,
                kurtosis,
                mask.float().mean(),
                grad_mag[mask].mean() if mask.any() else grad_mag.mean(),
                grad_mag[mask].std(unbiased=False) if mask.any() else grad_mag.std(unbiased=False),
                center_y,
                center_x,
                mask.float().sum() / float(mask.numel()),
                values.abs().mean(),
            ]
        )
        return features

    def forward(self, volume: torch.Tensor, voi_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch, depth, _, _ = volume.shape
        feature_slices = []
        for batch_index in range(batch):
            patient_features = []
            for slice_index in range(depth):
                mask = None if voi_mask is None else voi_mask[batch_index, slice_index]
                patient_features.append(self._single_slice_features(volume[batch_index, slice_index], mask))
            feature_slices.append(torch.stack(patient_features, dim=0))
        features = torch.stack(feature_slices, dim=0)
        tokens = self.projection(features)
        padding_mask = torch.zeros(batch, depth, device=volume.device, dtype=torch.bool)
        return tokens, padding_mask


@dataclass(slots=True)
class _TextCodes:
    section: int
    temporality: int
    negated: int


class MedBERTTextExpert(nn.Module):
    section_vocab = {
        "UNKNOWN": 0,
        "FINDINGS": 1,
        "IMPRESSION": 2,
        "HISTORY": 3,
        "ASSESSMENT": 4,
        "PLAN": 5,
    }
    temporality_vocab = {"past": 0, "present": 1, "future": 2}

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.text_vocab_size, config.embed_dim)
        self.section_embedding = nn.Embedding(len(self.section_vocab), config.embed_dim)
        self.temporality_embedding = nn.Embedding(len(self.temporality_vocab), config.embed_dim)
        self.negation_embedding = nn.Embedding(2, config.embed_dim)
        self.sentence_position_embedding = nn.Embedding(config.max_sentences, config.embed_dim)
        self.timestamp_embedding = nn.Embedding(config.max_timestamp_buckets, config.embed_dim)
        self.projection = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim),
        )

    def _tokenize(self, text: str) -> list[int]:
        pieces = [token.strip(".,:;()[]{}").lower() for token in text.split()]
        pieces = [token for token in pieces if token]
        if not pieces:
            pieces = ["[empty]"]
        return [_stable_hash(token, self.config.text_vocab_size) for token in pieces]

    def _codes(self, sentence: StructuredSentence) -> _TextCodes:
        section = self.section_vocab.get(sentence.section.upper(), self.section_vocab["UNKNOWN"])
        temporality = self.temporality_vocab.get(sentence.temporality.lower(), self.temporality_vocab["present"])
        return _TextCodes(section=section, temporality=temporality, negated=int(sentence.negated))

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime | None:
        if value is None:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None

    def _timestamp_bucket(
        self,
        sentence: StructuredSentence,
        reference_time: datetime | None,
    ) -> int:
        sentence_time = self._parse_timestamp(sentence.timestamp)
        if sentence_time is None or reference_time is None:
            return 0
        delta_days = max((sentence_time - reference_time).days, 0)
        return min(delta_days, self.config.max_timestamp_buckets - 1)

    def _encode_sentence(
        self,
        sentence: StructuredSentence,
        sentence_index: int,
        reference_time: datetime | None,
        device: torch.device,
    ) -> torch.Tensor:
        token_ids = torch.tensor(self._tokenize(sentence.text), device=device, dtype=torch.long)
        codes = self._codes(sentence)
        timestamp_bucket = self._timestamp_bucket(sentence, reference_time)
        base = self.token_embedding(token_ids).mean(dim=0)
        enriched = (
            base
            + self.section_embedding(torch.tensor(codes.section, device=device))
            + self.temporality_embedding(torch.tensor(codes.temporality, device=device))
            + self.negation_embedding(torch.tensor(codes.negated, device=device))
            + self.sentence_position_embedding(torch.tensor(min(sentence_index, self.config.max_sentences - 1), device=device))
            + self.timestamp_embedding(torch.tensor(timestamp_bucket, device=device))
        )
        return self.projection(enriched)

    def forward(self, batch_sentences: list[list[StructuredSentence]]) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.token_embedding.weight.device
        batch_size = len(batch_sentences)
        max_sentences = min(max((len(sentences) for sentences in batch_sentences), default=1), self.config.max_sentences)
        tokens = torch.zeros(batch_size, max_sentences, self.config.embed_dim, device=device)
        padding_mask = torch.ones(batch_size, max_sentences, device=device, dtype=torch.bool)
        for batch_index, sentences in enumerate(batch_sentences):
            truncated = sentences[:max_sentences]
            if not truncated:
                truncated = [StructuredSentence(text="[no clinical text provided]")]
            parsed_times = [self._parse_timestamp(sentence.timestamp) for sentence in truncated]
            reference_time = next((timestamp for timestamp in parsed_times if timestamp is not None), None)
            for token_index, sentence in enumerate(truncated):
                tokens[batch_index, token_index] = self._encode_sentence(
                    sentence=sentence,
                    sentence_index=token_index,
                    reference_time=reference_time,
                    device=device,
                )
                padding_mask[batch_index, token_index] = False
        return tokens, padding_mask


class ExpertFeatureExtractor(nn.Module):
    def __init__(self, config: MoEConfig | None = None) -> None:
        super().__init__()
        self.config = config or MoEConfig()
        dim = self.config.embed_dim
        self.medclip_pet = MedCLIPVisionExpert(embed_dim=dim)
        self.medclip_ct = MedCLIPVisionExpert(embed_dim=dim)
        self.medsam_pet = MedSAMVisionExpert(embed_dim=dim)
        self.medsam_ct = MedSAMVisionExpert(embed_dim=dim)
        self.radiomics_pet = RadiomicsExpert(config=self.config)
        self.radiomics_ct = RadiomicsExpert(config=self.config)
        self.text_expert = MedBERTTextExpert(config=self.config)
        self.image_position_embedding = nn.Parameter(torch.zeros(1, self.config.max_image_slices, dim))
        self.text_group_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        self.image_group_embedding = nn.ParameterDict(
            {
                "MedCLIP-PET": nn.Parameter(torch.zeros(1, 1, dim)),
                "MedCLIP-CT": nn.Parameter(torch.zeros(1, 1, dim)),
                "MedSAM-PET": nn.Parameter(torch.zeros(1, 1, dim)),
                "MedSAM-CT": nn.Parameter(torch.zeros(1, 1, dim)),
                "Radiomics-PET": nn.Parameter(torch.zeros(1, 1, dim)),
                "Radiomics-CT": nn.Parameter(torch.zeros(1, 1, dim)),
            }
        )

    def _default_pet_voi(self, pet_volume: torch.Tensor) -> torch.Tensor:
        return PETCTPreprocessor().threshold_41_percent_suvmax(pet_volume)

    def _prepare_voi_masks(
        self,
        pet_volume: torch.Tensor,
        pet_reference_volume: torch.Tensor | None,
        pet_voi: torch.Tensor | None,
        ct_voi: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pet_voi is None:
            reference_volume = pet_reference_volume if pet_reference_volume is not None else pet_volume
            pet_voi = self._default_pet_voi(reference_volume)
        if ct_voi is None:
            ct_voi = pet_voi
        return pet_voi.bool(), ct_voi.bool()

    def _add_image_position(self, tokens: torch.Tensor, group_name: str) -> torch.Tensor:
        slice_count = tokens.size(1)
        if slice_count > self.config.max_image_slices:
            raise ValueError(
                f"Slice count {slice_count} exceeds max_image_slices={self.config.max_image_slices}; "
                "increase the config to match the preprocessed axial depth."
            )
        return tokens + self.image_position_embedding[:, :slice_count] + self.image_group_embedding[group_name]

    def forward(
        self,
        pet_volume: torch.Tensor,
        ct_volume: torch.Tensor,
        text_batch: list[list[StructuredSentence]],
        pet_reference_volume: torch.Tensor | None = None,
        pet_voi: torch.Tensor | None = None,
        ct_voi: torch.Tensor | None = None,
    ) -> ExpertBatch:
        pet_voi, ct_voi = self._prepare_voi_masks(pet_volume, pet_reference_volume, pet_voi, ct_voi)
        pet_clip_tokens, pet_clip_mask = self.medclip_pet(pet_volume)
        ct_clip_tokens, ct_clip_mask = self.medclip_ct(ct_volume)
        pet_sam_tokens, pet_sam_mask = self.medsam_pet(pet_volume, pet_voi)
        ct_sam_tokens, ct_sam_mask = self.medsam_ct(ct_volume, ct_voi)
        pet_rad_tokens, pet_rad_mask = self.radiomics_pet(pet_volume, pet_voi)
        ct_rad_tokens, ct_rad_mask = self.radiomics_ct(ct_volume, ct_voi)
        text_tokens, text_mask = self.text_expert(text_batch)

        pet_clip_tokens = self._add_image_position(pet_clip_tokens, "MedCLIP-PET")
        ct_clip_tokens = self._add_image_position(ct_clip_tokens, "MedCLIP-CT")
        pet_sam_tokens = self._add_image_position(pet_sam_tokens, "MedSAM-PET")
        ct_sam_tokens = self._add_image_position(ct_sam_tokens, "MedSAM-CT")
        pet_rad_tokens = self._add_image_position(pet_rad_tokens, "Radiomics-PET")
        ct_rad_tokens = self._add_image_position(ct_rad_tokens, "Radiomics-CT")
        text_tokens = text_tokens + self.text_group_embedding

        group_tokens = {
            "MedCLIP-PET": pet_clip_tokens,
            "MedCLIP-CT": ct_clip_tokens,
            "MedSAM-PET": pet_sam_tokens,
            "MedSAM-CT": ct_sam_tokens,
            "Radiomics-PET": pet_rad_tokens,
            "Radiomics-CT": ct_rad_tokens,
            "MedBERT-Text": text_tokens,
        }
        group_padding_mask = {
            "MedCLIP-PET": pet_clip_mask,
            "MedCLIP-CT": ct_clip_mask,
            "MedSAM-PET": pet_sam_mask,
            "MedSAM-CT": ct_sam_mask,
            "Radiomics-PET": pet_rad_mask,
            "Radiomics-CT": ct_rad_mask,
            "MedBERT-Text": text_mask,
        }
        metadata = {"group_order": self.config.group_order, "pet_voi": pet_voi, "ct_voi": ct_voi}
        return ExpertBatch(group_tokens=group_tokens, group_padding_mask=group_padding_mask, metadata=metadata)
