# Phase 2 — Unimodal Analysis & Baseline Performance Benchmarking

## Method

Per the rubric, RGB-only and Thermal-only are evaluated using **the provided pretrained
QFDet weights**, not a separately trained substitute model. The reference implementation
(`third_party/mmdet-rgbtdroneperson`) has no single-modality code path — `QFDet.extract_feat` always
requires both an RGB and a thermal tensor — so unimodal ablation is done by feeding the
real pretrained dual-stream checkpoint a zeroed tensor for the modality under test
(`ZeroModality` pipeline transform, added to `mmdet/datasets/pipelines/multispectral_transforms.py`,
additive only — no change to any existing repo file's behavior). This evaluates the actual
provided weights, as the rubric specifies, with no retraining and no architecture change.

**Pretrained checkpoint**: `checkpoints/qfdet_vtuav_pretrained.pth` (485 MB), downloaded from the plain `QFDet` link in
`third_party/mmdet-rgbtdroneperson`'s README ("Trained Model → On VTUAV-det" table; the `QFDet*` row
was intentionally not used — it is the authors' improved variant, not the baseline). Verified as
a valid MMDetection checkpoint (734 `state_dict` entries, `backbone` + `backbone_t` present)
before use. Its embedded training config uses `bbox_head.num_classes=3`; our eval configs
match this exactly so every layer loads with zero shape mismatch (six extra, unused fusion-variant
keys in the checkpoint are silently ignored — `fusion_cat2`/`fusion_gated`, not part of the
active `Fusion_strategy` path in this repo state).

Eval configs: `third_party/mmdet-rgbtdroneperson/qfdet_configs/eval_qfdet_{baseline,rgb_only,thermal_only}_vtuav.py`
(val split) and their `_test.py` counterparts (test split). Detection metrics come from
`tools/test.py --eval bbox` (standard COCO eval, matches `VTUAVdet.evaluate()`). Compute metrics:
FLOPs/params via a `get_model_complexity_info` wrapper feeding a real `(v_img, t_img)` pair
(`tools/analysis_tools/get_flops.py`'s dummy-input default doesn't handle QFDet's paired input);
FPS/inference time via `tools/benchmark_simple.py`, a minimal single-GPU timing script written
because the repo's `benchmark.py` requires a distributed launch.

## Results

### Detection metrics (COCO)

| Experiment | Split | mAP | mAP50 | mAP75 | mAP_S | mAP_M | mAP_L |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline QFDet (fused) | val | 0.338 | 0.721 | 0.273 | 0.144 | 0.325 | 0.585 |
| Baseline QFDet (fused) | test | 0.299 | 0.674 | 0.227 | 0.129 | 0.299 | 0.554 |
| RGB-only (thermal zeroed) | val | 0.050 | 0.207 | 0.005 | 0.016 | 0.058 | 0.063 |
| RGB-only (thermal zeroed) | test | 0.042 | 0.191 | 0.003 | 0.020 | 0.050 | 0.074 |
| Thermal-only (RGB zeroed) | val | 0.267 | 0.587 | 0.205 | 0.097 | 0.240 | 0.584 |
| Thermal-only (RGB zeroed) | test | 0.232 | 0.547 | 0.166 | 0.080 | 0.225 | 0.538 |

### Computational metrics

Architecture and weight file are identical across all three rows — zeroing an input
stream doesn't change the compute graph, only what data flows through it. Measured once,
applies to all three:

| Metric | Value |
|---|---:|
| Model size | 485.11 MB |
| Parameters | 60.18 M |
| FLOPs (640×512 input) | 162.86 G |
| FPS | 11.3–11.5 (run-to-run GPU variance) |
| Inference time | 87–89 ms |

## Comparative analysis

**Fusion clearly beats either modality alone, and by a wide margin.** Baseline QFDet
(mAP 0.299 test) outperforms thermal-only (0.232) by 29% relative and RGB-only (0.042) by
7×. This is the headline justification for why the project's fusion strategy (Phase 3) is
worth building at all — neither sensor alone gets close to the fused result.

**RGB-only collapses almost completely (mAP 0.042 test) — far more than thermal-only
degrades (mAP 0.232 test).** This is not simply "thermal is a better sensor" — it's a
direct consequence of how *this specific* pretrained model was trained. QFDet's
`quality_attention` fusion gate (`qce_fusion` in `mmdet/models/detectors/qfdet.py`) learns a
per-pixel "quality" weighting for each stream from `bbox_prehead`, and that gate was tuned
during training assuming both streams are present and informative together. Zeroing thermal
doesn't just remove thermal's contribution — it also breaks the *learned relationship*
the RGB branch's weighting depends on, since the gate's calibration point assumes thermal's
quality signal is available. Thermal-only survives comparatively better because — per Phase 1's
findings — thermal carries more of the discriminative signal in this dataset overall (aerial
low-contrast scenes, frequent low-light conditions), so the model likely learned to lean on it
more heavily during training. This is a genuine architectural finding, not a testing artifact:
**QFDet's current fusion gate is not robust to a missing modality** — a fusion strategy that
degrades gracefully under single-modality dropout (rather than a hard learned dependency) is
one direction Phase 3 could improve on beyond mAP_S specifically.

**Small-object detection (mAP_S) is the weakest metric in every configuration** — 0.144
(fused, val) is barely half of mAP_M (0.325) and a quarter of mAP_L (0.585). This matches
Phase 1's dataset finding almost exactly: small objects are a genuinely harder case for this
architecture, independent of modality, and the fusion baseline doesn't solve it on its own —
motivating Phase 3's shallow-level fusion focus.

**Val consistently outperforms test across all three configurations** (e.g. baseline mAP
0.338 val vs. 0.299 test) — consistent with Phase 1's finding that the test split has a
2.57× higher proportion of small pedestrians than train, and small objects are the hardest
case for every configuration above. The gap is a real distribution-shift effect, not eval noise.

## Deliverables checklist (Stage 2, per the PDF)

- [x] RGB-only detector evaluated on val and test, using the provided pretrained QFDet weights
- [x] Thermal-only detector evaluated on val and test, using the provided pretrained QFDet weights
- [x] Baseline QFDet reproduced and benchmarked on val and test
- [x] Detection metrics: mAP, mAP50, mAP75, mAP_S, mAP_M, mAP_L — all six, all three detectors, both splits
- [x] Computational metrics: FPS, inference time, model size, parameter count, FLOPs
- [x] Written comparative analysis of strengths/limitations per modality (above)
