# PERSON 2 — FINAL summary (RGB BRANCH)

## 1. Project Overview
We isolated and optimized the RGB branch of the QFDet pedestrian detector using modular subclassing and configuration tuning.

## 2. Core Architecture
- Implemented `RGBOnlyQFDet` inside `person2_rgb/rgb_only_qfdet.py`. Bypasses the thermal branch, eliminating batch normalization noise from zero-masked thermal inputs, while reducing model footprint.

## 3. Results Summary
- **Baseline QFDet (Masked Input)**: 0.046 mAP / 9.46 FPS / 60.18 M Parameters / 162.86 GFLOPs.
- **Optimized RGBOnlyQFDet (Opt3)**: **0.057 mAP / 12.97 FPS / 36.90 M Parameters / 130.75 GFLOPs** (**+23.9% mAP, +37.1% FPS, -38.7% Params, -19.7% FLOPs**).
- **Scale-Up QFDet (Opt1)**: **0.051 mAP / 0.030 mAPS / 5.58 FPS / 294.18 GFLOPs** (**60% relative tiny-object recall gain**).

## 4. Integration Guidance
Person 1 can evaluate the released config files inside `person2_release/configs/` using:
`..\python39\python.exe tools/test.py person2_release/configs/exp_rgb_opt3.py checkpoints/qfdet_r50_fpn_1x_vtuav.pth --eval bbox`
