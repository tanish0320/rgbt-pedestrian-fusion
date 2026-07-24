# Future Improvements — RGB Branch (Person 2)

This document outlines realistic, evidence-based future improvements for the RGB-only
pedestrian detection branch of the QFDet RGB-Thermal pipeline. Every suggestion is
grounded in the experimental findings, failure modes, and architectural constraints
observed during this hackathon.

---

## 1. Tiny-Object Detection

### 1.1 Add a P2 FPN Level

**Current limitation:** The FPN outputs P3 (stride 8) as its highest-resolution level.
A 10×30 px pedestrian maps to only ~1.25×3.75 feature-map cells on P3 — barely
enough signal for the head to localize it.

**Proposed fix:** Add a P2 level (stride 4) using C2 features from the ResNet-50
backbone:

```python
# In the FPN config
neck = dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    start_level=0,          # was 1 (C3); change to 0 (C2)
    add_extra_convs='on_output',
    num_outs=5              # P2, P3, P4, P5, P6
)
```

**Expected gain:** Tiny pedestrians (area < 256 px²) gain 4× more feature-map cells.
Literature shows P2 integration consistently raises mAP_S by 3–8 points on drone
datasets.

**Trade-off:** P2 at stride 4 on a 640×512 input has spatial dimensions 160×128.
Processing this level adds ~30–40 GFLOPs and slows inference by ~15%.

---

### 1.2 Deformable Convolutions in the Backbone

**Current limitation:** Standard 3×3 convolutions sample a fixed rectangular grid.
Aerial pedestrians are often viewed at odd angles (leaning, mid-stride) and may not
align with fixed grid sampling.

**Proposed fix:** Replace the 3×3 convolutions in ResNet-50 stages 3–4 with
Deformable Conv v2 (DCNv2), which learns per-location spatial offsets:

```python
# In configs/person2/exp_rgb_dcn.py
model = dict(
    backbone=dict(
        type='ResNet',
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)  # apply to C4 and C5
    )
)
```

**Expected gain:** DCN is well-established for small-object detection; the backbone
learns to "reach" toward the pedestrian silhouette instead of sampling a fixed patch.

**Trade-off:** ~8–12% extra FLOPs; requires reloading the pretrained checkpoint and
fine-tuning the DCN offset layers (newly initialized) from the provided checkpoint.
No training-from-scratch violation, because only the DCN offset weights are new.

---

### 1.3 Test-Time Augmentation (TTA)

**Current limitation:** Single-scale, single-pass inference.

**Proposed fix:** At test time, run inference on multiple resolutions (e.g., 480×384,
640×512, 960×768) and merge predictions using Weighted Box Fusion (WBF):

```python
test_pipeline = [
    dict(type='MultiScaleFlipAug',
         img_scale=[(480, 384), (640, 512), (960, 768)],
         flip=True,
         transforms=[...])
]
```

**Expected gain:** TTA typically adds 1–3 mAP points with no model retraining.
The mAP_S gain can be larger (2–5 points) because the high-resolution pass gives
tiny objects the extra pixels they need.

**Trade-off:** 3–6× inference time. Not suitable for real-time deployment but valid
for hackathon final evaluation.

---

## 2. Confidence and Localization Quality

### 2.1 Soft-NMS Instead of Hard NMS

**Current limitation:** Hard NMS with IoU threshold 0.60 still suppresses some valid
crowd pedestrians whose boxes overlap by more than 60%.

**Proposed fix:** Replace hard NMS with Soft-NMS, which decays the confidence score
of overlapping boxes rather than deleting them outright:

```
Standard NMS:   score(box) = 0     if IoU(box, best) > threshold
Soft-NMS:       score(box) = score * e^(−IoU²/σ)
```

Boxes are suppressed only after their score falls below a minimum threshold, giving
legitimate overlapping pedestrians a chance to survive.

```python
model = dict(
    test_cfg=dict(
        nms=dict(type='soft_nms', iou_threshold=0.60, min_score=0.001)
    )
)
```

**Expected gain:** 0.3–1.0 mAP gain in crowd scenes with minimal compute overhead.

---

### 2.2 WBF: Weighted Box Fusion

**Current limitation:** NMS is a hard selection — it keeps exactly one box per
cluster, discarding all others even if they were accurate predictions from different
augmentation passes.

**Proposed fix:** Weighted Box Fusion averages the coordinates of overlapping
boxes weighted by their confidence scores. The fused box is often more accurate than
any individual prediction.

```python
from ensemble_boxes import weighted_boxes_fusion

boxes_list = [boxes_scale1, boxes_scale2, boxes_scale3]
scores_list = [scores1, scores2, scores3]
labels_list = [labels1, labels2, labels3]

fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
    boxes_list, scores_list, labels_list,
    iou_thr=0.55, skip_box_thr=0.001
)
```

**Best paired with TTA** (Section 1.3).

---

## 3. Backbone Improvements

### 3.1 Upgrade Backbone to ResNet-101 or Swin-T

**Current limitation:** ResNet-50 has a relatively small receptive field and limited
representational capacity for discriminating tiny pedestrians from background clutter.

**Option A — ResNet-101:**

```python
backbone=dict(type='ResNet', depth=101, ...)
```

Drop-in replacement. Adds ~23M parameters, ~40 GFLOPs, ~10% slower. Consistent
+0.5–1.5 mAP on small-object benchmarks.

**Option B — Swin Transformer Tiny:**

```python
backbone=dict(type='SwinTransformer', embed_dims=96, depths=[2,2,6,2], ...)
```

Transformer-based backbone with shifted-window attention. Better long-range context
(useful for recognizing partially occluded pedestrians). Requires loading Swin-T
ImageNet pretrained weights and fine-tuning — still satisfies the no-from-scratch
constraint.

**Trade-off:** Swin-T adds significant complexity (~28M params). Ensure MMCV version
supports it before switching.

---

### 3.2 CBAM: Channel and Spatial Attention in the Backbone

**Current limitation:** All feature channels are treated equally. Some channels
likely encode background texture (irrelevant) as strongly as pedestrian features.

**Proposed fix:** Insert CBAM (Convolutional Block Attention Module) after residual
blocks in Stage 3 and 4:

```
Channel Attention:  squeeze → FC → ReLU → FC → Sigmoid → channel weights
Spatial Attention:  avg+max pool along channels → Conv7×7 → Sigmoid → spatial weights
```

The module learns to suppress background channels and focus spatial attention on the
pedestrian region. Only the attention weights are new parameters — the backbone
weights load from the pretrained checkpoint.

---

## 4. Preprocessing Improvements

### 4.1 CLAHE Preprocessing for RGB Images

**Current limitation:** RGB images in the VTUAV-det dataset include low-contrast
scenes (shadows, dusk, overcast conditions) where pedestrians have low contrast
against the background.

**Proposed fix:** Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to
the L channel of the LAB color space during the test pipeline:

```python
import cv2
import numpy as np

def apply_clahe_rgb(image, clip_limit=2.0, tile_grid=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
```

Apply only to the L (luminance) channel to enhance contrast without altering color
hue — pedestrian clothing colors remain valid for the backbone.

**Expected gain:** Empirically 0.2–0.8 mAP on low-light subsets. Requires ablation
to confirm it doesn't increase false positives in uniform-background scenes.

---

### 4.2 Sparse Convolution at High Resolution

**Current limitation:** Processing 960×768 inputs costs 294 GFLOPs because standard
convolution computes at every pixel, including the >99% of pixels that are background.

**Proposed fix:** Use sparse convolutions (e.g., from `spconv` library) that only
compute activations at locations with non-trivial input. Pedestrian locations can be
estimated from a lightweight low-resolution pass first, then a sparse high-resolution
pass refines those predictions.

**Expected gain:** Reduce 960×768 FLOPs from ~294G to ~160–180G while preserving
the mAP_S gain of the scale-up experiment.

**Complexity:** High implementation effort. Requires custom data loaders and CUDA
kernel integration.

---

## 5. Training Improvements (if fine-tuning is permitted)

> All suggestions below involve fine-tuning from the provided pretrained checkpoint,
> not training from scratch. They comply with competition rules.

### 5.1 Mosaic Augmentation for Small Objects

**What it does:** Combines four training images into one, creating a more varied
scene with many small objects at different scales. Popularized by YOLOv4/v5.

```python
train_pipeline = [
    dict(type='Mosaic', img_scale=(640, 512), pad_val=114.0),
    dict(type='RandomAffine', scaling_ratio_range=(0.1, 2)),
    ...
]
```

**Expected gain:** Forces the model to learn small-object patterns from a diverse
spatial context. +1–3 mAP_S on drone datasets.

---

### 5.2 Copy-Paste Augmentation

**What it does:** During training, randomly paste pedestrian instances from one image
onto another. Artificially increases pedestrian density and creates novel occlusion
patterns.

**Expected gain:** Directly addresses the dataset imbalance (many background pixels,
few pedestrian pixels). Proven to improve mAP_S significantly on COCO small-object
benchmarks.

---

### 5.3 Focal Loss Tuning

**Current:** Default $\alpha=0.25$, $\gamma=2.0$ in QFDet's GFL head.

**Proposed:** Increase $\gamma$ to 3.0–4.0 to further down-weight the massive
number of easy background negatives and force the model to focus on hard small
pedestrian cases:

$$\mathcal{L}_{FL} = -\alpha(1-p_t)^\gamma \log(p_t)$$

At $\gamma=4$, an easy negative with $p_t=0.99$ contributes a loss of only
$-(0.01)^4 \approx 10^{-8}$ — effectively zero — while a hard positive with
$p_t=0.3$ contributes $-(0.7)^4 \times \log(0.3) \approx 0.179$. This sharpens
focus on the hard examples dramatically.

---

## 6. Fusion Integration Improvements

> These suggestions target the final RGB-Thermal fusion model (Person 1's scope)
> but are informed by Person 2's RGB branch findings.

### 6.1 Modality Dropout During Fusion Training

**Finding from RGB experiments:** The RGB-only model (mAP=0.055) is substantially
weaker than the full fusion model, but it is a valid unimodal predictor.

**Proposed:** During fusion fine-tuning, randomly drop the thermal branch with
probability $p=0.2$ per batch. This forces the model to maintain strong RGB-only
pathways, improving robustness when thermal sensors malfunction or are obstructed.

Our `RGBOnlyQFDet` clean zero-masking approach provides the correct implementation
template — zero out the **features** (not the input) to avoid BatchNorm contamination.

---

### 6.2 Scale-Aligned Feature Injection

**Finding:** Our 960×768 scale-up improved mAP_S by 60% (0.019 → 0.030). The fusion
model currently uses 640×512 for both modalities.

**Proposed:** Run the RGB backbone at 960×768 and thermal at 640×512, then align the
feature maps before fusion. The RGB branch benefits from higher resolution for tiny
targets while the thermal branch maintains its efficient resolution.

Spatial alignment can be achieved with a learnable 1×1 conv + bilinear interpolation
to match feature map dimensions before the fusion module.

---

## 7. Infrastructure and Reproducibility

### 7.1 ONNX Export for Deployment

Export `RGBOnlyQFDet` to ONNX format for deployment on edge devices (Jetson, FPGA):

```python
from mmdet.core.export import build_model_from_cfg, preprocess_example_input

input_config = {
    'input_shape': (1, 3, 512, 640),
    'input_name': 'input',
    'normalize_cfg': dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
}

torch.onnx.export(
    model,
    dummy_input,
    'person2_rgb_only.onnx',
    opset_version=11,
    do_constant_folding=True
)
```

Target: <100ms latency on Jetson Xavier NX for real-time drone deployment.

---

### 7.2 INT8 Quantization

Post-training quantization can reduce the model from float32 to int8 with minimal
accuracy loss (typically <0.5 mAP on well-calibrated models):

- Parameters: 36.90M float32 → ~9.2M int8 equivalent (4× smaller)
- Inference: 2–4× faster on hardware with INT8 support (most modern edge GPUs)

Use TensorRT or PyTorch's `torch.quantization` API with a small calibration dataset
(~200 images from the validation set).

---

## 8. Priority Matrix

| Improvement | Expected mAP Gain | Effort | Priority |
|:---|:---:|:---:|:---:|
| Add P2 FPN level | +1–3 mAP_S | Low | **High** |
| Soft-NMS | +0.3–1.0 mAP | Low | **High** |
| TTA | +1–3 mAP | Low | **High** |
| WBF | +0.5–1.5 mAP | Medium | **High** |
| DCNv2 backbone | +0.5–2.0 mAP | Medium | Medium |
| CLAHE preprocessing | +0.2–0.8 mAP | Low | Medium |
| Swin-T backbone | +1–3 mAP | High | Medium |
| Copy-paste augmentation | +1–3 mAP_S | Medium | Medium |
| Sparse convolution | Speed only | Very High | Low |
| INT8 quantization | Speed only | High | Low |

---

## 9. Summary

The three highest-priority zero-training improvements that can be applied immediately:

1. **Add P2 FPN level** — addresses the root cause of tiny pedestrian feature
   collapse with a single config change.

2. **Soft-NMS** — replaces hard suppression with score decay, improving crowd
   recall with no added parameters or training.

3. **Test-Time Augmentation + WBF** — leverages multi-scale predictions at
   inference time, directly targeting the mAP_S and mAP50 metrics that matter
   most in this competition.

All three are implementable through config changes alone, require no modification
to the pretrained checkpoint, and produce measurable gains validated by the
computer vision literature.

---

*Document authored by Person 2 (RGB Branch) — Yugma TechFest 2.0, MedhaDrishti.*
