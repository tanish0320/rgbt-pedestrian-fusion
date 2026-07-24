# Person 2 — RGB Branch Ablation Study

This document tabulates the step-by-step performance of the baseline RGB-only branch and our research-based optimizations on the validation (300 images) and test (200 images) splits.

## 1. Ablation Study Results

### Validation Split (300 images)

| Configuration | mAP | mAP50 | mAP75 | mAPS | mAPM | mAPL | FPS | Params | FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Masked-Input QFDet** | 0.053 | 0.219 | 0.005 | 0.018 | 0.061 | 0.075 | 9.56 | 60.18 M | 162.86 G |
| **RGBOnlyQFDet (Baseline)** | 0.072 | 0.278 | 0.011 | 0.017 | 0.080 | 0.130 | 9.60 | 36.90 M | 130.75 G |
| **+ Scale Up 960x768 (opt1)** | 0.056 | 0.243 | 0.005 | **0.028** | 0.065 | 0.091 | 5.40 | 36.90 M | 294.18 G |
| **+ NMS 0.45 (opt2)** | 0.070 | 0.275 | 0.011 | 0.016 | 0.078 | 0.128 | 9.60 | 36.90 M | 130.75 G |
| **+ NMS 0.60 (opt3) [Best Overall]** | **0.073** | 0.273 | 0.012 | 0.017 | 0.081 | 0.132 | 9.60 | 36.90 M | 130.75 G |

### Test Split (200 images)

| Configuration | mAP | mAP50 | mAP75 | mAPS | mAPM | mAPL | FPS | Params | FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Masked-Input QFDet** | 0.046 | 0.206 | 0.004 | 0.020 | 0.056 | 0.079 | 9.46 | 60.18 M | 162.86 G |
| **RGBOnlyQFDet (Baseline)** | 0.055 | 0.236 | 0.006 | 0.019 | 0.064 | 0.116 | 9.57 | 36.90 M | 130.75 G |
| **+ Scale Up 960x768 (opt1)** | 0.051 | 0.209 | 0.012 | **0.030** | 0.061 | 0.138 | 5.58 | 36.90 M | 294.18 G |
| **+ NMS 0.45 (opt2)** | 0.054 | 0.234 | 0.006 | 0.019 | 0.062 | 0.112 | 9.57 | 36.90 M | 130.75 G |
| **+ NMS 0.60 (opt3) [Best Overall]** | **0.057** | 0.232 | 0.007 | 0.019 | 0.065 | 0.120 | 9.57 | 36.90 M | 130.75 G |

---

## 2. Analysis of Results

- **Noise Reduction**: Bypassing the thermal backbone and returning clean zero-tensors (instead of passing zeros through BatchNorm layers) yielded a massive accuracy gain (val mAP from **0.053 to 0.072**, test mAP from **0.046 to 0.055**), while reducing params from **60.18 M to 36.9 M** and GFLOPs from **162.86 G to 130.75 G**.
- **Scale-Up Trade-off**: Scaling the input up to `960x768` (opt1) is extremely beneficial for tiny pedestrians, boosting $mAP_S$ from **0.019 to 0.030** (60% relative gain). However, it increases FLOPs from **130.75 G to 294.18 G** and lowers inference speed to ~5.5 FPS.
- **NMS Tuning**: Increasing NMS threshold to 0.60 (opt3) yields the best overall performance with no additional computational cost.
