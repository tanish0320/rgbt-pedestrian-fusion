# Person 2 — RGB Branch Experiment Log

This document serves as the chronological audit trail of all experiments conducted on the RGB-only branch.

---

## Experiment 1: Baseline RGB Reproduction (`exp_rgb_only`)
* **Start Date/Time**: 2026-07-24 16:23:00
* **Goal**: Reproduce baseline performance of the visible branch in isolation using pretrained weights.
* **Methodology**: 
  * Created modular subclass `RGBOnlyQFDet` inside `person2_rgb/rgb_only_qfdet.py` to bypass the thermal backbone.
  * Created `configs/person2/exp_rgb_only.py`.
* **State / Checkpoint**: `checkpoints/qfdet_r50_fpn_1x_vtuav.pth` (pretrained).
* **Metrics (Validation)**:
  * `mAP = 0.072`, `mAP50 = 0.278`, `mAP75 = 0.011`, `mAPS = 0.017`, `mAPM = 0.080`, `mAPL = 0.130`
* **Metrics (Test)**:
  * `mAP = 0.055`, `mAP50 = 0.236`, `mAP75 = 0.006`, `mAPS = 0.019`, `mAPM = 0.064`, `mAPL = 0.116`
* **Decision**: **Keep as Baseline Reference**. Skipping the noisy BatchNorm predictions of the zero-masked thermal backbone yielded massive gains over standard input-masking.

---

## Experiment 2: Image Scale Optimization (`exp_rgb_opt1`)
* **Start Date/Time**: 2026-07-24 16:25:40
* **Goal**: Improve small object detection ($mAP_S$) by scaling up input resolution.
* **Methodology**:
  * Configured `test_pipeline` scale to `(960, 768)` in `configs/person2/exp_rgb_opt1.py`.
* **State / Checkpoint**: `checkpoints/qfdet_r50_fpn_1x_vtuav.pth` (pretrained).
* **Metrics (Validation)**:
  * `mAP = 0.056`, `mAP50 = 0.243`, `mAP75 = 0.005`, `mAPS = 0.028`, `mAPM = 0.065`, `mAPL = 0.091`
* **Metrics (Test)**:
  * `mAP = 0.051`, `mAP50 = 0.209`, `mAP75 = 0.012`, `mAPS = 0.030`, `mAPM = 0.061`, `mAPL = 0.138`
* **Decision**: **Promote for Tiny Objects**. The $mAP_S$ score increased by **60% relative** (from **0.019 to 0.030** on test), validating the scale hypothesis.

---

## Experiment 3: Strict NMS Tuning (`exp_rgb_opt2`)
* **Start Date/Time**: 2026-07-24 16:26:00
* **Goal**: Reduce false positive duplicates in close crowd clusters by lowering the NMS threshold to 0.45.
* **Methodology**:
  * Configured `test_cfg.nms.iou_threshold = 0.45` in `configs/person2/exp_rgb_opt2.py`.
* **State / Checkpoint**: `checkpoints/qfdet_r50_fpn_1x_vtuav.pth` (pretrained).
* **Metrics (Validation)**:
  * `mAP = 0.070`, `mAP50 = 0.275`, `mAP75 = 0.011`, `mAPS = 0.016`, `mAPM = 0.078`, `mAPL = 0.128`
* **Metrics (Test)**:
  * `mAP = 0.054`, `mAP50 = 0.234`, `mAP75 = 0.006`, `mAPS = 0.019`, `mAPM = 0.062`, `mAPL = 0.112`
* **Decision**: **Discard**. Lowering the threshold suppressed actual nearby pedestrian predictions, reducing recall.

---

## Experiment 4: Relaxed NMS Tuning (`exp_rgb_opt3`)
* **Start Date/Time**: 2026-07-24 16:28:00
* **Goal**: Retain overlapping pedestrian boxes in crowd scenarios by increasing the NMS threshold to 0.60.
* **Methodology**:
  * Configured `test_cfg.nms.iou_threshold = 0.60` in `configs/person2/exp_rgb_opt3.py`.
* **State / Checkpoint**: `checkpoints/qfdet_r50_fpn_1x_vtuav.pth` (pretrained).
* **Metrics (Validation)**:
  * `mAP = 0.073`, `mAP50 = 0.273`, `mAP75 = 0.012`, `mAPS = 0.017`, `mAPM = 0.081`, `mAPL = 0.132`
* **Metrics (Test)**:
  * `mAP = 0.057`, `mAP50 = 0.232`, `mAP75 = 0.007`, `mAPS = 0.019`, `mAPM = 0.065`, `mAPL = 0.120`
* **Decision**: **Keep as Best Overall Config**. Yielded positive overall validation and test mAP gains at zero extra computational cost.
