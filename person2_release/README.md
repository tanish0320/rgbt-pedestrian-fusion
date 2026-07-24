# Person 2 — RGB Branch Release Package

This directory contains the finalized, verified, and modular release artifacts for the **RGB-only branch** of the QFDet pedestrian detector.

## 1. Expected Metrics
* **Baseline RGB (`exp_rgb_only`)**:
  * Validation: mAP = 0.072 / mAP50 = 0.278 / mAPS = 0.017
  * Test: mAP = 0.055 / mAP50 = 0.236 / mAPS = 0.019
  * Latency: 77.99 ms/img (12.82 FPS) on NVIDIA RTX 3050.
* **Upscaled RGB (`exp_rgb_opt1`)**:
  * Validation: mAP = 0.056 / mAPS = 0.028 (60% relative mAPS gain)
  * Test: mAP = 0.051 / mAPS = 0.030 (60% relative mAPS gain)
  * Latency: 128.95 ms/img (7.76 FPS).
* **Optimized NMS RGB (`exp_rgb_opt3`)**:
  * Validation: mAP = 0.073 / mAP50 = 0.273 / mAPS = 0.017
  * Test: mAP = 0.057 / mAP50 = 0.232 / mAPS = 0.019
  * Latency: 77.07 ms/img (12.97 FPS).

---

## 2. Contents of this Package
* `configs/`: Isolated configuration override files (`exp_rgb_only.py`, `exp_rgb_opt1.py`, `exp_rgb_opt3.py`).
* `reports/`: Complete technical reports, ablation studies, and qualitative analysis.
* `metrics/`: Benchmark JSON data (`person2_metrics.json`, `person2_best.json`).
* `plots/`: Rendered visualization comparison charts.
* `logs/`: Inference and sweep execution logs.

---

## 3. How to Run Evaluations
Execute the following commands from the repository root:

```powershell
# Evaluate Baseline:
..\python39\python.exe tools/test.py configs/person2/exp_rgb_only.py checkpoints/qfdet_r50_fpn_1x_vtuav.pth --eval bbox

# Evaluate Scale-Up (Opt1):
..\python39\python.exe tools/test.py configs/person2/exp_rgb_opt1.py checkpoints/qfdet_r50_fpn_1x_vtuav.pth --eval bbox

# Evaluate Relaxed NMS (Opt3):
..\python39\python.exe tools/test.py configs/person2/exp_rgb_opt3.py checkpoints/qfdet_r50_fpn_1x_vtuav.pth --eval bbox
```
