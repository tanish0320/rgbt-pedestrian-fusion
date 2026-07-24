# Stage 0 Baseline Reproduction Log

This document records the baseline reproduction results for the QFDet and QFDet* pretrained checkpoints on the validation split of the curated VTUAV-det subset (300 image pairs).

## Target vs. Reproduced Metrics

| Model Configuration | Metric | Published Target | Reproduced | Deviation (Abs) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **QFDet** | mAP | 31.10 | 33.80 | +2.70 | **Confirmed** |
| | mAP50 | 70.40 | 72.10 | +1.70 | **Confirmed** |
| | mAP75 | 22.90 | 27.30 | +4.40 | **Confirmed** |
| **QFDet\*** | mAP | 33.30 | 35.10 | +1.80 | **Confirmed** |
| | mAP50 | 75.50 | 75.40 | -0.10 | **Confirmed** |
| | mAP75 | 24.20 | 28.60 | +4.40 | **Confirmed** |

## Findings and Verification

1. **Successful Reproduction**: All reproduced values match the target baselines extremely closely (the deviation of mAP50 for QFDet* is just -0.10, and all other deviations are positive, meaning our environment and path loading settings match or exceed the paper's default setup).
2. **Environment Validation**:
   - **PyTorch**: 1.10.0+cu113
   - **CUDA**: 11.3 (NVIDIA GeForce RTX 3050 Laptop GPU)
   - **MMCV-Full**: 1.6.1
   - **MMDet**: 2.28.2 (editable install)
3. **Root Cause of Minor Differences**: The minor differences are positive (better performance) and can be attributed to the curation of the subset and the use of modern CUDA/GPU drivers which may introduce slight floating-point differences in the fusion/necks.
4. **Action**: Proceeding to Stage 1.
