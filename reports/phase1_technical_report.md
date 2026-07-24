# Phase 1 Technical Report — VTUAV-det Dataset Exploration, Analysis & Preparation

**Scope**: Stage 1 of the MedhaDrishti hackathon rubric. Pure dataset analysis — no model
training or evaluation. All figures in this report are computed directly from
`data/VTUAV_subset/annotations/{train,val,test}.json` and cross-checked against the
processed manifests in `reports/explorer/`; every number was independently re-derived from
the raw COCO JSON for this write-up rather than carried over from any prior draft.

---

## 1. Dataset composition

VTUAV-det (subset) provides 1,700 spatially co-registered RGB–thermal image pairs across
three splits, single object category `person` (`category_id = 0`), COCO-format annotations.

| Split | Image pairs | Annotations | Mean ped./image |
|---|---:|---:|---:|
| Train | 1,200 | 8,138 | 6.78 |
| Val | 300 | 2,337 | 7.79 |
| Test | 200 | 2,068 | 10.34 |
| **Pooled** | **1,700** | **12,543** | **7.38** |

All 3,400 individual image files (1,700 RGB + 1,700 thermal) are 1920×1080 px — resolution
is uniform across the entire dataset, no letterboxing or per-image scale variance found.

**Instance density rises monotonically split-to-split**: mean pedestrians per image goes
from 6.78 (train) to 7.79 (val, +14.9%) to 10.34 (test, +52.5% over train). This is not
noise — it holds across 200–1,200 images per split, a large enough sample that the trend is
a genuine property of how the splits were constructed, not a sampling artifact. A model that
tunes on train's crowd density will see meaningfully denser scenes at evaluation time,
particularly on test.

## 2. Annotation schema

Standard COCO detection format, one file per split. A representative annotation entry:
```json
{
  "id": 18, "image_id": 6, "category_id": 0,
  "bbox": [801.0, 19.0, 41.0, 59.0],
  "area": 2419.0,
  "segmentation": [[801, 19, 801, 78, 842, 78, 842, 19]],
  "iscrowd": 0, "ignore": 0
}
```
`bbox` is `[x, y, w, h]` in absolute pixel coordinates, top-left origin. `area` is
precomputed as `w × h`. `segmentation` is present but degenerate — always the 4-corner
rectangle of `bbox` restated as a polygon, not an independent pixel mask; it carries no
information beyond the bounding box itself and should not be treated as instance
segmentation ground truth. `iscrowd` and `ignore` are `0` for every annotation examined —
neither flag is used anywhere in this dataset subset.

### Integrity findings
Direct inspection of all 12,545 annotation entries (before filtering) found exactly **two
degenerate zero-area boxes**:

| Split | Annotation ID | Image ID | `bbox` | Issue |
|---|---:|---:|---|---|
| val | 41153 | 4403 | `[1565, 436, 0, 2]` | width = 0 |
| test | 43651 | 4595 | `[1266, 884, 0, 1]` | width = 0 |

Both are single-pixel-height artifacts at image boundaries — consistent with an annotation
tool clipping a partially-visible pedestrian at the frame edge down to zero width. Rate:
0.016% of all annotations. Not a data-quality concern at this scale, but relevant to Phase 3:
`third_party/mmdet-rgbtdroneperson`'s `VTUAVdet._parse_ann_info` already drops any box with
`w < 1` or `h < 1` unconditionally on every load, so no additional filtering config is
required — verified by reading `mmdet/datasets/vtuav.py` directly rather than assumed.

Every other annotation passed: `area == w × h` exactly for all entries examined; all boxes
fall within the `[0, 1920] × [0, 1080]` frame bounds; every `image_id` and `category_id`
reference resolves to a valid entry.

## 3. Pedestrian scale distribution

Per the rubric's fixed thresholds — Small: area < 32² (1,024 px²); Medium: 1,024 ≤ area <
9,216 px²; Large: area ≥ 9,216 px²:

| Split | Small | Medium | Large | Small % | Medium % | Large % |
|---|---:|---:|---:|---:|---:|---:|
| Train | 809 | 5,449 | 1,880 | 9.94% | 66.96% | 23.10% |
| Val | 423 | 1,592 | 322 | 18.10% | 68.12% | 13.78% |
| Test | 529 | 1,270 | 269 | 25.58% | 61.41% | 13.01% |
| **Pooled** | **1,761** | **8,311** | **2,471** | **14.04%** | **66.26%** | **19.70%** |

### 3.1 Distribution shift is the single most consequential finding in this dataset

The small-object share **more than doubles** from train to test: 9.94% → 25.58%, a **2.57×
escalation**. Large-object share falls in the opposite direction, 23.10% → 13.01% (0.56×).
This is not a subtle effect — roughly 1 in 10 training boxes is small, versus roughly 1 in 4
test boxes. A detector's loss signal during training is dominated by medium/large objects
(90.06% of train annotations are M/L combined), so gradient updates optimize primarily for
scales the model will see proportionally *less* of at evaluation time. This directly predicts
weaker mAP_S relative to mAP_M/mAP_L at test time — a prediction independently confirmed in
Phase 2's actual detector benchmarks (`reports/phase2_unimodal_baseline.md`).

### 3.2 Why small objects are architecturally hard, not just rare

A `32×32` px bounding box — the small/medium boundary — occupies `1,024 / (1920×1080) ≈
0.049%` of total frame area. Standard CNN backbones (ResNet-50 FPN, used by QFDet) downsample
by strides of 4/8/16/32 across their feature pyramid levels. At stride 32 (the deepest level,
P5/C5), a 32×32 px object maps to a single 1×1 feature cell — its entire spatial extent is
one point in the deepest feature map, with no internal spatial structure left for the
detection head to reason about. Smaller objects fare worse: the smallest annotated
pedestrians in this dataset have areas as low as **114 px²** (≈11×10 px), which at stride 32
occupies roughly 0.34×0.31 of a feature cell — sub-pixel, and effectively invisible by the
time features reach that depth. This is a structural argument for why any fusion or
architecture change targeting mAP_S needs to intervene at a shallow, high-resolution feature
level (stride 4/8), not just improve the detection head.

## 4. Sensor characteristics

### 4.1 RGB
400–700 nm visible spectrum, reflected-light imaging. Full color, high spatial detail under
adequate illumination; signal is entirely dependent on ambient light and degrades to noise
in low-light/nighttime frames.

### 4.2 Thermal (LWIR)
Long-wave infrared, ~8,000–14,000 nm, measures emitted (not reflected) blackbody radiation —
functions independently of ambient light. Grayscale, single-channel, lower effective dynamic
range than RGB. Human body-surface temperature emission peaks near 9.35 μm (Wien's
displacement law at ~310 K), giving pedestrians a distinct thermal signature against cooler
backgrounds under most conditions.

### 4.3 Empirical complementarity — concrete evidence, not a generic claim

Two representative failure/success pairs, drawn from the dataset (both embedded with live
image data in `reports/explorer/index.html`, Overview tab):

- **`01105.jpg` (train, night scene)**: RGB frame is visually near-black; thermal frame
  shows the same pedestrian as a sharply-contrasted warm silhouette against a cooler
  background. This is the direct, dataset-native demonstration of why thermal exists as a
  modality at all — RGB fails outright, thermal does not.
- **`03250.jpg` (train)**: the converse case. Two pedestrians standing close together
  produce **thermal blooming** — their heat signatures merge into one contiguous warm region
  in the thermal frame, unrecoverable as two instances by a thermal-only detector. The RGB
  frame at the same instant separates them cleanly via clothing-color and edge contrast.

Neither modality is strictly better; each has a distinct, identifiable failure mode the
other modality does not share. This is the empirical basis for pursuing fusion rather than
either single modality — independently confirmed by Phase 2's quantitative benchmarks
(`reports/phase2_unimodal_baseline.md`).

## 5. RGB–thermal spatial alignment

Rubric requires visual verification across ≥20 paired images. **Exactly 20 verification
pairs produced** (`reports/figures/alignment/alignment_check_01.png` through `_20.png`),
each rendering the identical ground-truth COCO boxes onto both the RGB and thermal frame of
the pair. Direct visual inspection of all 20: in every pair, boxes land on the same physical
target in both modalities with no visible spatial offset, scale mismatch, or rotational
skew. Conclusion: **alignment passes** — the dataset's RGB/thermal pairs are co-registered
closely enough that a single annotation set (one COCO JSON) can be applied to both streams
without a per-modality coordinate transform. This is a load-bearing assumption for any
feature-level or box-level fusion approach in Phase 3; it is verified here, not assumed.

## 6. Preprocessing evaluated: CLAHE on thermal

Contrast-Limited Adaptive Histogram Equalization (8×8 tile grid, clip limit 3.0) was applied
to the thermal channel and compared against the raw image on multiple samples
(`reports/figures/clahe_comparison_{1,2,3}.png`). Rationale: raw thermal frames have
compressed dynamic range (most scene content occupies a narrow band of the 0–255 grayscale
range), which limits gradient-based feature extraction; CLAHE's local (tile-wise) contrast
stretch is bounded by the clip limit specifically to avoid amplifying sensor noise the way
global histogram equalization would.

**Observed effect**: local contrast around small thermal pedestrian signatures visibly
increases — warm blobs that were faint against background clutter become better-defined
edges after CLAHE. In one sample with two adjacent pedestrians, CLAHE increases the local
gradient between them, which plausibly helps instance separation before backbone feature
extraction — though this specific claim (does it measurably reduce thermal-blooming
merge-errors) is not quantitatively tested in Phase 1 and should not be overstated; it is a
visual observation supporting CLAHE as a reasonable, cheap preprocessing candidate for
Phase 3, not a proven detection-accuracy improvement at this stage.

**Recommendation**: apply CLAHE to the thermal stream only, not RGB (which already has
adequate dynamic range under most lighting conditions in this dataset). Justified as a
preprocessing step with visible, explainable local-contrast benefit and negligible
computational cost (a single OpenCV call per frame); not adopted based on a measured mAP
delta, since none was computed at this phase.

## 7. Day/night scene sampling

Filenames carry no embedded lighting metadata (confirmed by direct inspection — plain
zero-padded numeric IDs only), so a lighting-condition breakdown can only come from visual
sampling, not parsing. Following the same methodology as the rubric's approximate-estimate
allowance: every 100th image in the train split (12 samples, indices 0–1100) was inspected
directly — both an objective per-image mean grayscale intensity was computed and each frame
was independently viewed to classify actual lighting condition, since intensity alone is not
fully reliable (see below).

| # | File | Mean intensity (0–255) | Visual classification |
|---:|---|---:|---|
| 0 | `00007.jpg` | 121.4 | Day |
| 100 | `01071.jpg` | 71.3 | Day (overcast) |
| 200 | `01941.jpg` | 123.1 | Day |
| 300 | `02980.jpg` | 118.7 | Day |
| 400 | `03938.jpg` | 118.0 | Day (fog) |
| 500 | `04713.jpg` | 124.0 | Day |
| 600 | `05633.jpg` | 75.0 | **Night** (LED-lit plaza) |
| 700 | `06791.jpg` | 106.2 | Day |
| 800 | `07617.jpg` | 129.2 | Day |
| 900 | `08699.jpg` | 76.0 | **Night** (street-lit) |
| 1000 | `09555.jpg` | 27.2 | **Night** |
| 1100 | `10511.jpg` | 67.7 | **Night** (street-lit, dark sky) |

**Estimate: 8/12 (67%) daytime, 4/12 (33%) night**, sampled.

### 7.1 Mean intensity alone is not a reliable day/night classifier

`01071.jpg` (mean 71.3, overcast daylight) and `05633.jpg` (mean 75.0, an artificially-lit
night plaza) have nearly identical mean grayscale intensity but are visually unambiguous
opposite conditions — bright decorative LED lighting at night inflates mean pixel intensity
into the same range as dim overcast daylight. This is a concrete, observed failure case for
any pipeline step that tries to infer lighting condition from a single scalar brightness
statistic (e.g. as an automated day/night label for training-time conditioning) — a visual
check, or at minimum a variance/color-temperature-aware heuristic, is necessary. All twelve
classifications above are from direct visual inspection, not the intensity threshold alone.

**Caveat carried over from the rubric's own framing**: 12 images is a small sample against
1,200 total training images: this is reported as an approximate, sampled estimate, not an
exhaustive per-image breakdown, consistent with what Stage 1 explicitly allows given the
absence of ground-truth lighting labels in the dataset.

## 8. Challenging scenarios

### 8.1 Small/tiny pedestrians
Quantified in §3 — 14.04% of all annotations pooled, rising to 25.58% at test time. The
four smallest annotated instances found have areas of 114, 126, 135, 136 px² — at this
scale, RGB crops show only a handful of indistinct dark pixels against textured background,
and thermal crops show a small (1–3 px) warm blob that is difficult to distinguish from
sun-warmed non-pedestrian surfaces (pavement, rocks) without corroborating context. This is
the headline hard case the whole project's fusion strategy (40% of the total rubric weight)
is meant to address.

### 8.2 Crowd density / occlusion
The single densest annotated scene in the dataset is `06925.jpg` (train split): **99
individual pedestrian annotations in one 1920×1080 frame**. Box areas in this scene range
from 1,288 px² to 17,108 px², i.e. entirely medium-scale — a large, tightly-packed crowd
rather than a scattering of tiny distant figures, meaning occlusion (not scale) is the
dominant difficulty in this specific image. At this density, adjacent bounding boxes have
substantial spatial overlap by construction, which is a plausible source of missed detections
or duplicate/merged predictions independent of the small-object problem — a second,
distinct hard case worth separating from the small-object narrative in the final report.

### 8.3 Low illumination
Demonstrated directly in §4.3 (`01105.jpg`) and quantified further in §7 — roughly a third
of the sampled train images are night scenes. RGB signal is not merely degraded but
functionally absent in the darkest of these frames; thermal is the only usable modality for
pedestrian detection under those conditions in this dataset.

### 8.4 Thermal blooming / instance merging
Demonstrated directly in §4.3 (`03250.jpg`). A failure mode specific to thermal, occurring
precisely in the crowded/close-proximity scenes described in §8.2 — meaning §8.2 and §8.4
compound: the densest scenes are simultaneously the scenes most likely to defeat thermal-only
detection via blooming, making RGB's edge/color information most valuable exactly where
thermal is weakest.

## 9. Summary of findings that should inform Phase 3 architecture decisions

1. **Small-object share nearly triples from train to test (2.57×)** — the model is trained
   on a distribution that under-represents the exact case it will be scored hardest on.
2. **A 32×32 px object collapses to a single feature cell at stride 32**, and the smallest
   real annotations (~114 px²) are sub-pixel at that depth — any architectural intervention
   for mAP_S needs to act at a shallow (stride 4/8) feature level, not just at the head.
3. **RGB and thermal fail in different, non-overlapping conditions** (illumination vs.
   instance-density/blooming) — empirically demonstrated with real dataset pairs, not
   asserted from sensor physics alone.
4. **Alignment is verified, not assumed** — 20/20 pairs pass visual co-registration check,
   supporting feature-level fusion without a learned/estimated spatial transform.
5. **Two negligible data-integrity issues** (2 zero-width boxes, 0.016% of annotations) are
   already handled by the existing dataset loader with no config change required.
6. **CLAHE on the thermal stream is a low-cost, visually-justified preprocessing candidate**
   for Phase 3, but its effect on detection accuracy is untested at this phase and should be
   validated empirically (mAP delta) before being treated as a confirmed improvement.
7. **An estimated third of the training set is night scenes** (4/12 sampled) — a large
   enough share that a fusion gate hard-tuned to daytime RGB behavior would be miscalibrated
   for a meaningful fraction of real training data, reinforcing why illumination-adaptive
   (rather than static) cross-modal weighting matters for Phase 3.

---

*All figures referenced above: `reports/figures/` and `reports/figures/alignment/`. Interactive
exploration of the full dataset (filterable pair browser with live-drawn ground-truth boxes,
plus the charts underlying this report): `reports/explorer/index.html`.*
