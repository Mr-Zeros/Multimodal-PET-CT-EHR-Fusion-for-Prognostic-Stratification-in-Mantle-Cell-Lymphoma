from __future__ import annotations

import re
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .config import PreprocessingConfig
from .data import PETMetrics, StructuredSentence


class PETCTPreprocessor:
    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()

    def to_suv(
        self,
        pet_volume: torch.Tensor,
        injected_dose_mbq: float | None = None,
        body_weight_kg: float | None = None,
        decay_correction: float = 1.0,
    ) -> torch.Tensor:
        if injected_dose_mbq is None or body_weight_kg is None:
            return pet_volume
        suv_scale = (body_weight_kg * 1000.0) / max(injected_dose_mbq * decay_correction, self.config.epsilon)
        return pet_volume * suv_scale

    def robust_body_mask(self, volume: torch.Tensor) -> torch.Tensor:
        positive = volume[volume > 0]
        if positive.numel() == 0:
            return torch.ones_like(volume, dtype=torch.bool)
        threshold = torch.quantile(positive, self.config.body_mask_percentile / 100.0)
        return volume >= threshold

    def normalize_pet(self, suv_volume: torch.Tensor) -> torch.Tensor:
        lower, upper = self.config.pet_clip_range
        suv_volume = suv_volume.clamp(lower, upper)
        mask = self.robust_body_mask(suv_volume)
        values = suv_volume[mask]
        mean = values.mean() if values.numel() > 0 else suv_volume.mean()
        std = values.std(unbiased=False) if values.numel() > 1 else suv_volume.std(unbiased=False)
        std = std.clamp_min(self.config.epsilon)
        return (suv_volume - mean) / std

    def normalize_ct(self, ct_volume: torch.Tensor) -> torch.Tensor:
        lower, upper = self.config.ct_clip_range
        ct_volume = ct_volume.clamp(lower, upper)
        return 2.0 * (ct_volume - lower) / (upper - lower) - 1.0

    def threshold_41_percent_suvmax(self, pet_volume: torch.Tensor, body_mask: torch.Tensor | None = None) -> torch.Tensor:
        if pet_volume.ndim == 3:
            pet_volume = pet_volume.unsqueeze(0)
        if body_mask is None:
            body_mask = pet_volume > 0
        body_mask = body_mask.bool()
        masked_pet = torch.where(body_mask, pet_volume, torch.zeros_like(pet_volume))
        suvmax = masked_pet.flatten(start_dim=1).amax(dim=1)
        threshold = (self.config.voi_threshold_fraction * suvmax).view(-1, 1, 1, 1)
        voi = (pet_volume >= threshold) & body_mask
        return voi[0] if voi.shape[0] == 1 else voi

    def compute_pet_metrics(
        self,
        pet_volume: torch.Tensor,
        voi_mask: torch.Tensor,
        voxel_spacing_mm: Sequence[float] | None = None,
    ) -> PETMetrics:
        if pet_volume.ndim == 3:
            pet_volume = pet_volume.unsqueeze(0)
        if voi_mask.ndim == 3:
            voi_mask = voi_mask.unsqueeze(0)
        voxel_spacing_mm = tuple(voxel_spacing_mm or self.config.default_voxel_spacing_mm)
        voxel_volume_cm3 = float(voxel_spacing_mm[0] * voxel_spacing_mm[1] * voxel_spacing_mm[2]) / 1000.0
        mask = voi_mask.bool()
        flat_pet = pet_volume.flatten(start_dim=1)
        flat_mask = mask.flatten(start_dim=1)
        suvmax = []
        suvmean = []
        tmtv = []
        tlg = []
        for patient_pet, patient_mask in zip(flat_pet, flat_mask, strict=True):
            values = patient_pet[patient_mask]
            if values.numel() == 0:
                values = patient_pet
                voxel_count = torch.tensor(0.0, device=patient_pet.device)
            else:
                voxel_count = patient_mask.float().sum()
            patient_suvmax = values.max()
            patient_suvmean = values.mean()
            patient_tmtv = voxel_count * voxel_volume_cm3
            patient_tlg = patient_suvmean * patient_tmtv
            suvmax.append(patient_suvmax)
            suvmean.append(patient_suvmean)
            tmtv.append(patient_tmtv)
            tlg.append(patient_tlg)
        return PETMetrics(
            suvmax=torch.stack(suvmax),
            suvmean=torch.stack(suvmean),
            tmtv=torch.stack(tmtv),
            tlg=torch.stack(tlg),
        )

    @staticmethod
    def _ensure_5d(volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim == 3:
            return volume.unsqueeze(0).unsqueeze(0)
        if volume.ndim == 4:
            return volume.unsqueeze(1)
        if volume.ndim == 5:
            return volume
        raise ValueError(f"Expected 3D, 4D, or 5D tensor, got {tuple(volume.shape)}")

    def resample_volume(self, volume: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
        volume_5d = self._ensure_5d(volume)
        resampled = F.interpolate(volume_5d, size=target_shape, mode="trilinear", align_corners=False)
        if volume.ndim == 3:
            return resampled[0, 0]
        if volume.ndim == 4:
            return resampled[:, 0]
        return resampled

    def _soft_histogram(self, values: torch.Tensor, bins: int) -> torch.Tensor:
        mins = values.amin(dim=1, keepdim=True)
        maxs = values.amax(dim=1, keepdim=True)
        spans = (maxs - mins).clamp_min(self.config.epsilon)
        centers = torch.linspace(0.0, 1.0, bins, device=values.device, dtype=values.dtype)
        normalized = (values - mins) / spans
        sigma = 1.0 / max(bins, 1)
        weights = torch.exp(-0.5 * ((normalized.unsqueeze(-1) - centers) / sigma) ** 2)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.config.epsilon)
        return weights

    def normalized_mutual_information(self, x: torch.Tensor, y: torch.Tensor, bins: int | None = None) -> torch.Tensor:
        bins = bins or self.config.registration_bins
        x_flat = x.flatten(start_dim=1)
        y_flat = y.flatten(start_dim=1)
        x_weights = self._soft_histogram(x_flat, bins)
        y_weights = self._soft_histogram(y_flat, bins)
        px = x_weights.mean(dim=1).clamp_min(self.config.epsilon)
        py = y_weights.mean(dim=1).clamp_min(self.config.epsilon)
        joint = torch.einsum("bni,bnj->bij", x_weights, y_weights)
        joint = joint / joint.sum(dim=(1, 2), keepdim=True).clamp_min(self.config.epsilon)
        joint = joint.clamp_min(self.config.epsilon)
        hx = -(px * px.log()).sum(dim=-1)
        hy = -(py * py.log()).sum(dim=-1)
        hxy = -(joint * joint.log()).sum(dim=(1, 2))
        return (hx + hy) / hxy.clamp_min(self.config.epsilon)

    def register_pet_to_ct(self, pet_volume: torch.Tensor, ct_volume: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pet_small = self.resample_volume(pet_volume, self.config.registration_shape)
        ct_small = self.resample_volume(ct_volume, self.config.registration_shape)
        pet_small_5d = self._ensure_5d(pet_small)
        ct_small_5d = self._ensure_5d(ct_small)

        batch_size = pet_small_5d.shape[0]
        device = pet_small_5d.device
        dtype = pet_small_5d.dtype
        identity = torch.eye(3, 4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        theta = identity.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([theta], lr=self.config.registration_lr)

        for _ in range(self.config.registration_steps):
            grid = F.affine_grid(theta, size=ct_small_5d.shape, align_corners=False)
            moved = F.grid_sample(pet_small_5d, grid, mode="bilinear", padding_mode="border", align_corners=False)
            nmi = self.normalized_mutual_information(moved[:, 0], ct_small_5d[:, 0]).mean()
            reg = (theta - identity).square().mean()
            loss = -nmi + self.config.registration_regularization * reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        registered = self.apply_affine_transform(pet_volume, ct_volume, theta.detach())
        if pet_volume.ndim == 3:
            return registered[0, 0], theta.detach()[0]
        if pet_volume.ndim == 4:
            return registered[:, 0], theta.detach()
        return registered, theta.detach()

    def apply_affine_transform(
        self,
        moving_volume: torch.Tensor,
        reference_volume: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        moving_5d = self._ensure_5d(moving_volume)
        reference_5d = self._ensure_5d(reference_volume)
        if theta.ndim == 2:
            theta = theta.unsqueeze(0)
        grid = F.affine_grid(theta, size=reference_5d.shape, align_corners=False)
        return F.grid_sample(moving_5d, grid, mode="bilinear", padding_mode="border", align_corners=False)

    def preprocess_pair(
        self,
        pet_volume: torch.Tensor,
        ct_volume: torch.Tensor,
        injected_dose_mbq: float | None = None,
        body_weight_kg: float | None = None,
        build_default_voi: bool = False,
    ) -> dict[str, torch.Tensor]:
        pet_resampled = self.resample_volume(pet_volume, self.config.target_shape)
        ct_resampled = self.resample_volume(ct_volume, self.config.target_shape)
        pet_suv = self.to_suv(pet_resampled, injected_dose_mbq, body_weight_kg)
        pet_norm = self.normalize_pet(pet_suv)
        ct_norm = self.normalize_ct(ct_resampled)
        pet_registered, affine = self.register_pet_to_ct(pet_norm, ct_norm)
        pet_suv_registered = self.apply_affine_transform(pet_suv, ct_norm, affine)[0, 0]
        outputs = {"pet": pet_registered, "pet_suv": pet_suv_registered, "ct": ct_norm, "affine": affine}
        if build_default_voi:
            voi = self.threshold_41_percent_suvmax(pet_suv_registered)
            outputs["pet_voi"] = voi
            outputs["ct_voi"] = voi
        return outputs


class EHRTextPreprocessor:
    section_pattern = re.compile(r"^\s*([A-Za-z][A-Za-z /\-]{1,40})\s*:\s*$")
    sentence_pattern = re.compile(r"(?<=[.!?;])\s+|\n+")
    negation_pattern = re.compile(r"\b(no|not|without|negative for|denies|free of)\b", re.IGNORECASE)
    past_pattern = re.compile(r"\b(history of|previous|prior|was|were|had)\b", re.IGNORECASE)
    future_pattern = re.compile(r"\b(plan|recommend|will|follow-up|next)\b", re.IGNORECASE)

    def deidentify(self, report: str) -> str:
        report = re.sub(r"\b\d{6,}\b", "[ID]", report)
        report = re.sub(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b", "[DATE]", report)
        report = re.sub(r"\b(?:MRN|PATIENT|NAME)\s*[:#]?\s*[A-Za-z0-9_-]+\b", "[IDENTIFIER]", report, flags=re.IGNORECASE)
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in report.splitlines()]
        return "\n".join(line for line in lines if line).strip()

    def temporality(self, sentence: str) -> str:
        if self.future_pattern.search(sentence):
            return "future"
        if self.past_pattern.search(sentence):
            return "past"
        return "present"

    def structure_report(self, report: str, timestamp: str | None = None) -> list[StructuredSentence]:
        cleaned = self.deidentify(report)
        chunks = [chunk.strip() for chunk in self.sentence_pattern.split(cleaned) if chunk.strip()]
        current_section = "UNKNOWN"
        structured: list[StructuredSentence] = []
        for chunk in chunks:
            match = self.section_pattern.match(chunk)
            if match:
                current_section = match.group(1).upper()
                continue
            structured.append(
                StructuredSentence(
                    text=chunk,
                    section=current_section,
                    negated=bool(self.negation_pattern.search(chunk)),
                    temporality=self.temporality(chunk),
                    timestamp=timestamp,
                )
            )
        return structured

    def structure_batch(self, reports: list[str], timestamps: list[str | None] | None = None) -> list[list[StructuredSentence]]:
        timestamps = timestamps or [None] * len(reports)
        return [self.structure_report(report, timestamp) for report, timestamp in zip(reports, timestamps, strict=True)]
