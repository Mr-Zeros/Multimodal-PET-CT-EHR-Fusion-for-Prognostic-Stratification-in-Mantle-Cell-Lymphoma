<div align="center">

# Interpretable Multimodal PET/CT-EHR MoE for MCL

**PyTorch implementation for interpretable prognostic modeling in mantle cell lymphoma**

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-111111?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-CB2B1E?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Multimodal-PET%2FCT%20%2B%20EHR-0F766E?style=for-the-badge" alt="Multimodal">
  <img src="https://img.shields.io/badge/Fusion-Mixture%20of%20Experts-1D4ED8?style=for-the-badge" alt="MoE">
  <img src="https://img.shields.io/badge/Interpretability-Attention%20%2B%20Gating-7C3AED?style=for-the-badge" alt="Interpretability">
</p>

</div>

---

## Project Overview

This repository presents the implementation of the multimodal framework:

> **Interpretable Multimodal PET/CT-EHR Fusion via Mixture-of-Experts for Prognostic Stratification in Mantle Cell Lymphoma**

- PET/CT preprocessing with harmonization and registration
- EHR structuring into sentence-level, time-aware tokens
- Seven expert groups spanning PET, CT, radiomics, and clinical text
- Two-stage hierarchical attention-based **mixture-of-experts (MoE)** fusion
- Independent **PFS** and **OS** Cox-style survival heads
- Interpretable outputs including attention rollout, gating contributions, and modality ablation
- Downstream **multiparametric models** combining R-signatures with PET and clinical factors

---


An interpretable framework is proposed that jointly models:

- **PET** for metabolic heterogeneity
- **CT** for morphology and structural context
- **EHR text** for clinically meaningful semantic information

Most public examples of multimodal oncology modeling either use simple concatenation or omit interpretability. This repository instead demonstrates a full **hierarchical expert fusion pipeline** with explicit intermediate representations and clinician-facing interpretability hooks.

---

## Architecture at a Glance

```mermaid
flowchart LR
    A[Baseline PET] --> B[PET/CT Harmonization]
    C[Baseline CT] --> B
    D[Clinical Notes / Reports] --> E[EHR Structuring]

    B --> F1[MedCLIP-PET]
    B --> F2[MedCLIP-CT]
    B --> F3[MedSAM-PET]
    B --> F4[MedSAM-CT]
    B --> F5[Radiomics-PET]
    B --> F6[Radiomics-CT]
    E --> F7[MedBERT-Text]

    F1 --> G[Intra-group Transformer + Attention Pooling]
    F2 --> G
    F3 --> G
    F4 --> G
    F5 --> G
    F6 --> G
    F7 --> G

    G --> H[Cross-group Transformer]
    H --> I[Gating Network]
    I --> J[Patient-level Decision Embedding]

    J --> K1[PFS Cox Head]
    J --> K2[OS Cox Head]

    J --> L1[Attention Rollout Heatmaps]
    J --> L2[Expert / Modality Contributions]
    J --> L3[Multiparametric Models]
```

---

## Design

### 1. Multimodal preprocessing

- PET and CT are resampled to a shared voxel grid
- PET is represented in SUV-style intensity space and normalized for downstream learning
- PET is aligned to CT through affine registration driven by normalized mutual information
- Clinical text is de-identified, segmented, section-tagged, negation-tagged, and time-aware

### 2. Seven expert groups

seven expert-specific feature groups:

| Group | Input | Intended Role |
|---|---|---|
| `MedCLIP-PET` | PET axial slices | high-level metabolic semantics |
| `MedCLIP-CT` | CT axial slices | high-level anatomic semantics |
| `MedSAM-PET` | PET + lesion-aware cues | morphology-sensitive PET representation |
| `MedSAM-CT` | CT + lesion-aware cues | morphology-sensitive CT representation |
| `Radiomics-PET` | PET VOI slices | handcrafted metabolic features |
| `Radiomics-CT` | CT VOI slices | handcrafted structural features |
| `MedBERT-Text` | structured clinical sentences | text semantics and clinical context |

### 3. Hierarchical MoE fusion

The fusion module follows the two-stage structure:

- **Intra-group aggregation**: each expert group is contextualized by a lightweight Transformer and pooled by learned attention
- **Inter-group mixture**: refined group vectors interact through a second Transformer, then a gating network produces adaptive expert weights

### 4. Survival modeling

Two independent endpoint-specific models are included:

- **PFS model**
- **OS model**

Each endpoint has its own fusion stack and linear Cox-style risk head.

### 5. Multiparametric modeling

Learned R-signatures with PET and clinical factors are further combined.

- **PFS multiparametric model**: `R-signature + TLG + WBC + Ki-67`
- **OS multiparametric model**: `R-signature + TLG + β2-microglobulin`

---

## Interpretability

Interpretability is treated as a first-class output rather than an afterthought.

The current implementation provides:

- **Attention rollout** for slice-level importance maps
- **Volume heatmaps** projected back onto PET/CT
- **Inter-group gating weights** to quantify expert contribution
- **Modality-level contribution summaries** for PET, CT, and EHR
- **Expert ablation** to inspect performance sensitivity
- **Subtype-linked risk summaries** for histopathologic interpretation

This design reflects emphasis on clinically coherent and biologically meaningful explanations.
