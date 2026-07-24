# Ablation Summary — Report Asset

This asset compiles the publication-quality comparison tables for validation and test splits:

### Standalone RGB Branch Ablation Ladder

```markdown
Validation Split (300 images):
- Masked-Input QFDet Baseline:  0.053 mAP / 0.018 mAPS / 9.56 FPS
- RGBOnlyQFDet (Clean Masking): 0.072 mAP / 0.017 mAPS / 9.60 FPS  <-- (+35.8% mAP improvement)
- + Scale Up 960x768 (opt1):    0.056 mAP / 0.028 mAPS / 5.40 FPS  <-- (Best Tiny Object Recall)
- + NMS 0.60 (opt3):            0.073 mAP / 0.017 mAPS / 9.60 FPS  <-- (Best Overall mAP)

Test Split (200 images):
- Masked-Input QFDet Baseline:  0.046 mAP / 0.020 mAPS / 9.46 FPS
- RGBOnlyQFDet (Clean Masking): 0.055 mAP / 0.019 mAPS / 9.57 FPS  <-- (+19.5% mAP improvement)
- + Scale Up 960x768 (opt1):    0.051 mAP / 0.030 mAPS / 5.58 FPS  <-- (60% relative mAPS gain)
- + NMS 0.60 (opt3):            0.057 mAP / 0.019 mAPS / 9.57 FPS  <-- (+23.9% mAP improvement)
```
