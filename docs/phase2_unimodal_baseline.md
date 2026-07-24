# Person 2 & 3 — Unimodal Baseline (Phase 2, 15%) — DONE

Originally split as two roles: Person 2 owning RGB-only, Person 3 owning Thermal-only, each
training a separate model on their own GPU. That plan is superseded — see below for why —
and Phase 2 is complete as a single unified effort.

## Why the original plan changed

The PDF requires RGB-only and Thermal-only evaluated using **the provided pretrained QFDet
weights**, not a separately trained model per modality (training from scratch is explicitly
disallowed). The original plan — Person 2 and Person 3 each training a standalone
torchvision `fasterrcnn_resnet50_fpn_v2` on their own images-only subset — didn't actually
satisfy that: it produced two working detectors, but neither used the provided pretrained
weights, so it didn't meet the rubric's literal requirement. Those runs were deleted (see
`STATUS.md` notes log) once the compliant approach below was built, to avoid the two sets of
numbers being confused later.

## What Phase 2 actually does

The reference `QFDet` architecture (`third_party/mmdet-rgbtdroneperson`) has no
single-modality code path — `QFDet.extract_feat` always requires both an RGB and thermal
tensor. So RGB-only and Thermal-only are evaluated by feeding the **real pretrained
dual-stream checkpoint** a zeroed tensor for the modality under test (`ZeroModality`
pipeline transform, added to
`third_party/mmdet-rgbtdroneperson/mmdet/datasets/pipelines/multispectral_transforms.py`).
This evaluates the actual provided weights with one input ablated — matching the rubric —
rather than substituting a differently-trained model. Baseline QFDet (both streams real) is
evaluated the same way with no ablation.

## Results

Full methodology, results table (6 detection + 5 computational metrics × 3 detectors × 2
splits), and written comparative analysis: **`reports/phase2_unimodal_baseline.md`**.
Metrics also in `results/metrics.csv` (`exp_baseline_qfdet_*`, `exp_rgb_only_qfdet_*`,
`exp_thermal_only_qfdet_*` rows).

Headline: Baseline QFDet (mAP 0.299 test) beats Thermal-only (0.232) by 29% and RGB-only
(0.042) by 7× — the clearest evidence for why Phase 3's fusion strategy is worth building.

## Reproducing

```bash
# from repo root, with the qfdet conda env active and PYTHONPATH set (see README.md)
cd third_party/mmdet-rgbtdroneperson
python tools/test.py qfdet_configs/eval_qfdet_baseline_vtuav.py ../../checkpoints/qfdet_vtuav_pretrained.pth --eval bbox
python tools/test.py qfdet_configs/eval_qfdet_rgb_only_vtuav.py ../../checkpoints/qfdet_vtuav_pretrained.pth --eval bbox
python tools/test.py qfdet_configs/eval_qfdet_thermal_only_vtuav.py ../../checkpoints/qfdet_vtuav_pretrained.pth --eval bbox
# append _test to any config name above for the test-split numbers instead of val
```

## Handoff

Phase 2 complete — checkpoint (`checkpoints/qfdet_vtuav_pretrained.pth`) and its baseline
metrics feed directly into Phase 3 (fine-tuning starting point, per
`docs/YOU_person1_fusion.md`) and Phase 4 (comparison table).
