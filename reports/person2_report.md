# Person 2 — RGB Branch Research & Optimization Report

This report presents the findings, baselines, and research-based optimization results for the **RGB-only Branch** of the QFDet architecture. All developments have been completed independently from the fusion branch using modular subclassing and dynamic configuration overrides.

## 1. Repository Inspection and Modality Isolation
- **Baseline Modality Mismatch**: The standard QFDet configuration processes dual-modality pairs by running both ResNet-50 backbones. If one modality is zero-masked at input, its features undergo normalization (BatchNorm) layers in eval mode, introducing noise and degrading final predictions.
- **Modularity Solution**: We implemented `RGBOnlyQFDet` inside `person2_rgb/rgb_only_qfdet.py` inheriting from `QFDet`. It completely bypasses the thermal backbone forward pass and sets thermal features to clean zero-tensors. This reduces memory footprint and avoids BatchNorm noise from the masked branch.
- **Dynamic Registration**: Registered subclass using MMDetection's `custom_imports` hook, allowing standard tools (`test.py`, `get_flops.py`) to run it without edits to the core registry files.

## 2. Complexity Analysis
Using `get_flops.py` at input size `640x512`, we compare the RGB-only branch to the standard QFDet baseline:

| Architecture | Parameters (M) | FLOPs (GFLOPs) | Save relative to Fusion |
| :--- | :--- | :--- | :--- |
| **QFDet (Fusion)** | 60.18 M | 162.86 G | - |
| **RGBOnlyQFDet (Ours)** | 36.90 M | 130.75 G | **38.7% Params / 19.7% FLOPs saved** |

---

## 3. Optimization Hypotheses & Findings

### Experiment 1 (`exp_rgb_opt1`): Scale Up to 960x768
- **Hypothesis**: Scaling up input images increases the spatial resolution of tiny pedestrians (area < 256), allowing FPN fine scale levels (P3/P4) to extract higher-quality features and improve tiny pedestrian recall ($mAP_S$).
- **Result**: **Highly Successful**. While overall mAP dropped slightly due to scale mismatch on large objects, $mAP_S$ increased by **60% relative** (from **0.017 to 0.028** on val, and **0.019 to 0.030** on test). 

### Experiment 2 (`exp_rgb_opt2`): NMS Threshold Tuned to 0.45
- **Hypothesis**: Lowering NMS threshold suppresses false positives in dense pedestrian clusters.
- **Result**: **Unsuccessful**. mAP decreased slightly, showing that a strict NMS threshold suppresses actual overlapping pedestrians.

### Experiment 3 (`exp_rgb_opt3`): NMS Threshold Tuned to 0.60
- **Hypothesis**: Increasing NMS threshold retains highly overlapping predictions, improving overall pedestrian recall.
- **Result**: **Successful**. Overall mAP increased to **0.073** on val and **0.057** on test split, representing a solid gain with zero additional compute cost.

---

## 4. Handoff Strategy
Our best RGB-only weights and config (`exp_rgb_opt3` for overall mAP, `exp_rgb_opt1` for tiny object recall) are completely isolated in `configs/person2/` and `person2_rgb/` and can be directly referenced by Person 1 to optimize the RGB branch within the final fusion model.
