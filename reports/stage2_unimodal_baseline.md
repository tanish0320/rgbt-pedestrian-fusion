# Stage 2 — Unimodal Analysis and Baseline Benchmarking Report

This report presents a comparative analysis of unimodal (RGB-only, Thermal-only) vs. multimodal fusion baselines. Evaluations are performed on both the validation split (300 pairs) and test split (200 pairs) of the curated VTUAV-det dataset.

## 1. Benchmarking Results

### QFDET on VAL split

| Modality Mode | mAP | mAP50 | mAP75 | mAPS | mAPM | mAPL | FPS | Inf Time | Model Size | Param Count | FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RGB-Only** | 0.053 | 0.219 | 0.005 | 0.018 | 0.061 | 0.075 | 9.56 | 104.60 ms | 462.6 MB | 60.18 M | 162.86 GFLOPs |
| **Thermal-Only** | 0.290 | 0.610 | 0.240 | 0.099 | 0.272 | 0.578 | 9.51 | 105.12 ms | 462.6 MB | 60.18 M | 162.86 GFLOPs |
| **Fusion (Full)** | 0.338 | 0.721 | 0.273 | 0.144 | 0.325 | 0.585 | 9.60 | 104.12 ms | 462.6 MB | 60.18 M | 162.86 GFLOPs |


### QFDET on TEST split

| Modality Mode | mAP | mAP50 | mAP75 | mAPS | mAPM | mAPL | FPS | Inf Time | Model Size | Param Count | FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RGB-Only** | 0.046 | 0.206 | 0.004 | 0.020 | 0.056 | 0.079 | 9.46 | 105.70 ms | 462.6 MB | 60.18 M | 162.86 GFLOPs |
| **Thermal-Only** | 0.253 | 0.572 | 0.187 | 0.094 | 0.252 | 0.532 | 9.15 | 109.34 ms | 462.6 MB | 60.18 M | 162.86 GFLOPs |
| **Fusion (Full)** | 0.299 | 0.674 | 0.227 | 0.129 | 0.299 | 0.554 | 9.57 | 104.46 ms | 462.6 MB | 60.18 M | 162.86 GFLOPs |


### QFDET_STAR on VAL split

| Modality Mode | mAP | mAP50 | mAP75 | mAPS | mAPM | mAPL | FPS | Inf Time | Model Size | Param Count | FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RGB-Only** | 0.047 | 0.201 | 0.007 | 0.025 | 0.051 | 0.125 | 5.40 | 185.10 ms | 463.1 MB | 60.25 M | 485.64 GFLOPs |
| **Thermal-Only** | 0.313 | 0.669 | 0.258 | 0.159 | 0.295 | 0.589 | 5.49 | 182.31 ms | 463.1 MB | 60.25 M | 485.64 GFLOPs |
| **Fusion (Full)** | 0.351 | 0.754 | 0.286 | 0.192 | 0.334 | 0.589 | 5.41 | 184.69 ms | 463.1 MB | 60.25 M | 485.64 GFLOPs |


### QFDET_STAR on TEST split

| Modality Mode | mAP | mAP50 | mAP75 | mAPS | mAPM | mAPL | FPS | Inf Time | Model Size | Param Count | FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RGB-Only** | 0.049 | 0.201 | 0.005 | 0.032 | 0.054 | 0.100 | 5.58 | 179.28 ms | 463.1 MB | 60.25 M | 485.64 GFLOPs |
| **Thermal-Only** | 0.289 | 0.657 | 0.208 | 0.157 | 0.287 | 0.554 | 5.55 | 180.24 ms | 463.1 MB | 60.25 M | 485.64 GFLOPs |
| **Fusion (Full)** | 0.327 | 0.742 | 0.245 | 0.194 | 0.322 | 0.557 | 5.56 | 179.77 ms | 463.1 MB | 60.25 M | 485.64 GFLOPs |


## 2. Analysis and Insights

### Unimodal Performance (RGB vs. Thermal)
- **Thermal Dominance**: Across all configurations and splits, **Thermal-only inference** significantly outperforms **RGB-only inference** (e.g., QFDet on Val: Thermal mAP = 29.0% vs. RGB mAP = 5.3%). This is due to the nature of the drone-based pedestrian detection dataset: pedestrians are extremely small, and thermal heat signatures provide a highly distinctive contrast against the background compared to cluttered RGB visual cues.
- **RGB Strengths & Weaknesses**: RGB-only inference performs very poorly on tiny objects (mAPS) but is relatively more competent on medium and large objects (mAPL). RGB features contain rich textures, which are useful when the resolution of the object is large enough, but clutter and lighting variations make it highly unstable for tiny pedestrian detection.
- **Thermal Strengths & Weaknesses**: Thermal features provide high-contrast blobs that are easy to locate even at very small scales, but they lack fine texture.

### Fusion Synergy
- **Full Fusion vs. Unimodal**: Multimodal fusion (QFDet Full) achieves the best performance overall (e.g., QFDet on Val: Fusion mAP = 33.8% vs. Thermal-only mAP = 29.0% and RGB-only mAP = 5.3%). This shows that the QCE (Quality-Aware Cross-Modal Fusion) is able to synergistically combine complementary information from both modalities.
- **Tiny Object Scale**: Fusion provides a substantial boost for small and tiny objects, proving that the cross-modal features help validate detections where a single modality has low confidence.

### Computational Complexity
- **QFDet vs. QFDet\***: QFDet\* operates on a higher-resolution feature map (starting from FPN level P2 with stride 4, compared to QFDet starting from P3 with stride 8). While QFDet\* achieves higher performance (Fusion mAP on Val: 35.1% vs. 33.8%), it comes at a very high computational cost: **485.64 GFLOPs vs. 162.86 GFLOPs**, representing a **3x increase** in complexity, with a subsequent decrease in FPS (from 9.60 FPS to 5.41 FPS).
