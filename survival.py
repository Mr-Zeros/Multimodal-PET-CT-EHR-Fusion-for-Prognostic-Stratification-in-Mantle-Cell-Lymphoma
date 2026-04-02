from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig
from .data import EndpointOutput, ExpertBatch
from .experts import ExpertFeatureExtractor
from .fusion import HierarchicalMoEFusion


def cox_partial_log_likelihood(risk_score: torch.Tensor, event_time: torch.Tensor, event_indicator: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(event_time, descending=True)
    risk_score = risk_score[order]
    event_indicator = event_indicator[order].float()
    log_risk_set = torch.logcumsumexp(risk_score, dim=0)
    observed = event_indicator > 0
    if observed.sum() == 0:
        return risk_score.new_tensor(0.0)
    partial_log_likelihood = risk_score[observed] - log_risk_set[observed]
    return -partial_log_likelihood.mean()


class EndpointTaskModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fusion = HierarchicalMoEFusion(config.moe)
        self.risk_head = nn.Linear(config.moe.embed_dim, 1)

    def forward(self, expert_batch: ExpertBatch) -> EndpointOutput:
        fusion_output = self.fusion(expert_batch)
        risk = self.risk_head(fusion_output.patient_embedding).squeeze(-1)
        return EndpointOutput(risk=risk, r_signature=risk, fusion=fusion_output)


class MultitaskMCLSurvivalModel(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.shared_extractor = ExpertFeatureExtractor(self.config.moe)
        self.endpoint_models = nn.ModuleDict(
            {
                "pfs": EndpointTaskModel(self.config),
                "os": EndpointTaskModel(self.config),
            }
        )

    def extract_experts(
        self,
        pet_volume: torch.Tensor,
        ct_volume: torch.Tensor,
        text_batch,
        pet_reference_volume: torch.Tensor | None = None,
        pet_voi: torch.Tensor | None = None,
        ct_voi: torch.Tensor | None = None,
    ) -> ExpertBatch:
        return self.shared_extractor(
            pet_volume=pet_volume,
            ct_volume=ct_volume,
            text_batch=text_batch,
            pet_reference_volume=pet_reference_volume,
            pet_voi=pet_voi,
            ct_voi=ct_voi,
        )

    def forward(
        self,
        pet_volume: torch.Tensor,
        ct_volume: torch.Tensor,
        text_batch,
        pet_reference_volume: torch.Tensor | None = None,
        pet_voi: torch.Tensor | None = None,
        ct_voi: torch.Tensor | None = None,
    ) -> tuple[dict[str, EndpointOutput], ExpertBatch]:
        expert_batch = self.extract_experts(
            pet_volume=pet_volume,
            ct_volume=ct_volume,
            text_batch=text_batch,
            pet_reference_volume=pet_reference_volume,
            pet_voi=pet_voi,
            ct_voi=ct_voi,
        )
        outputs = {endpoint: model(expert_batch) for endpoint, model in self.endpoint_models.items()}
        return outputs, expert_batch

    def losses(
        self,
        outputs: dict[str, EndpointOutput],
        targets: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        losses = {}
        for endpoint, endpoint_output in outputs.items():
            endpoint_target = targets[endpoint]
            losses[endpoint] = cox_partial_log_likelihood(
                risk_score=endpoint_output.risk,
                event_time=endpoint_target["time"],
                event_indicator=endpoint_target["event"],
            )
        losses["total"] = sum(losses.values())
        return losses
