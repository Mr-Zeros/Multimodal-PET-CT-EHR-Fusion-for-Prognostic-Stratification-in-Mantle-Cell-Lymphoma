from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .data import ClinicalCovariates, PETMetrics


@dataclass(slots=True)
class MultiparametricOutput:
    endpoint: str
    feature_names: tuple[str, ...]
    features: torch.Tensor
    risk: torch.Tensor


class EndpointMultiparametricModel(nn.Module):
    def __init__(self, endpoint: str) -> None:
        super().__init__()
        if endpoint not in {"pfs", "os"}:
            raise ValueError(f"Unsupported endpoint: {endpoint}")
        self.endpoint = endpoint
        self.feature_names = (
            ("r_signature", "log_tlg", "wbc_elevated", "ki67_high")
            if endpoint == "pfs"
            else ("r_signature", "log_tlg", "beta2_microglobulin_elevated")
        )
        self.linear = nn.Linear(len(self.feature_names), 1)

    def _build_features(
        self,
        r_signature: torch.Tensor,
        clinical: ClinicalCovariates,
    ) -> torch.Tensor:
        log_tlg = torch.log1p(clinical.tlg.float())
        components = [r_signature.float(), log_tlg]
        if self.endpoint == "pfs":
            if clinical.wbc_elevated is None or clinical.ki67_high is None:
                raise ValueError("PFS multiparametric model requires WBC and Ki-67 covariates")
            components.extend([clinical.wbc_elevated.float(), clinical.ki67_high.float()])
        else:
            if clinical.beta2_microglobulin_elevated is None:
                raise ValueError("OS multiparametric model requires beta2-microglobulin covariate")
            components.append(clinical.beta2_microglobulin_elevated.float())
        return torch.stack(components, dim=-1)

    def forward(self, r_signature: torch.Tensor, clinical: ClinicalCovariates) -> MultiparametricOutput:
        features = self._build_features(r_signature, clinical)
        risk = self.linear(features).squeeze(-1)
        return MultiparametricOutput(
            endpoint=self.endpoint,
            feature_names=self.feature_names,
            features=features,
            risk=risk,
        )


class MultiparametricSurvivalSuite(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.models = nn.ModuleDict(
            {
                "pfs": EndpointMultiparametricModel("pfs"),
                "os": EndpointMultiparametricModel("os"),
            }
        )

    @staticmethod
    def clinical_from_metrics(
        pet_metrics: PETMetrics,
        wbc_elevated: torch.Tensor | None = None,
        ki67_high: torch.Tensor | None = None,
        beta2_microglobulin_elevated: torch.Tensor | None = None,
    ) -> ClinicalCovariates:
        return ClinicalCovariates(
            tlg=pet_metrics.tlg,
            wbc_elevated=wbc_elevated,
            ki67_high=ki67_high,
            beta2_microglobulin_elevated=beta2_microglobulin_elevated,
        )

    def forward(
        self,
        pfs_r_signature: torch.Tensor,
        os_r_signature: torch.Tensor,
        clinical: ClinicalCovariates,
    ) -> dict[str, MultiparametricOutput]:
        return {
            "pfs": self.models["pfs"](pfs_r_signature, clinical),
            "os": self.models["os"](os_r_signature, clinical),
        }
