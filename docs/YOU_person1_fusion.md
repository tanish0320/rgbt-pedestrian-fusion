# Person 1 (You) — Fusion Lead — local RTX 4060

The 40%-weighted core stage. Strategy locked in, main config built and smoke-tested — this
doc now covers running the real training and folding in the two ablation runs from Person 2
and Person 3.

## Environment — already resolved
conda env `qfdet` (torch 1.10.0+cu113, mmcv-full 1.6.1, mmdet 2.25.1) confirmed working with
GPU detection. `numpy<2` / `opencv-python<4.10` pinned (avoids a real numpy-2.x/torch-1.10
compatibility break). See `README.md` Setup for the full from-scratch install path.

## Strategy: P2 fusion + modality dropout
Full rationale in `docs/phase3_fusion_plan.md`. Two changes to the baseline QFDet, both
grounded in Phase 1/2 findings, not a generic idea:
1. **P2/stride-4 FPN level** — the backbone already computes C2, the baseline config just
   discards it (`start_level=1`). Targets Phase 1's finding that a 32×32px box collapses to
   a single feature cell at stride 32, and the smallest real annotations (~114px²) are
   already sub-pixel by then.
2. **Modality-dropout training** (`RandomMasking`, already existed in the repo, unused —
   reused rather than writing new code) — targets Phase 2's finding that RGB-only ablation
   collapsed 7× (mAP 0.042) vs. thermal-only's 29% degradation, evidence the existing
   `quality_attention` gate was never trained under a missing/weak-modality condition.

## Your config — built and smoke-tested
`third_party/mmdet-rgbtdroneperson/qfdet_configs/qfdet_r50_fpn_p2_vtuav.py` — full docstring
explains every change and the checkpoint-load mismatches you'll see on startup (3
`lateral_convs` + 2 `atss_cls` reinitialize, expected and already analyzed — not a bug).

**Smoke-tested** (5 real train steps, `tools/smoke_test_p2.py`): peak VRAM 4,446MB/8.59GB,
~0.8s/step after warmup, loss dropping 178.9→33.6 over 5 steps. Estimated ~50min for a full
6-epoch run at 600 iters/epoch.

## Run the full training
```powershell
cd third_party/mmdet-rgbtdroneperson
$env:PYTHONPATH = (Get-Location).Path
python tools/train.py qfdet_configs/qfdet_r50_fpn_p2_vtuav.py
```
Checkpoint saves to `checkpoints/train_qfdet_fusion_v1/` per epoch. Save periodically /
resume from last epoch if interrupted — same as any long local run.

## Evaluate
```powershell
python tools/test.py qfdet_configs/qfdet_r50_fpn_p2_vtuav.py checkpoints/train_qfdet_fusion_v1/latest.pth --eval bbox
```
For test-split numbers, mirror Phase 2's `_test.py` pattern (point `data.test` at
`test.json`/`mmdet_data/test/`). Also run `tools/benchmark_simple.py` for FPS/inference-time
and the `get_model_complexity_info` wrapper for FLOPs/params (same pattern as
`reports/phase2_unimodal_baseline.md`'s Method section) — expect FLOPs higher than the
162.86G baseline, that's the real cost of P2, not an error.

Add results to `results/metrics.csv`: `exp_fusion_v1_val`, `exp_fusion_v1_test`.

## Ablation study — Person 2 and Person 3
Two isolation runs on the 3050s, splitting `fusion_v1`'s two changes apart:
- **Person 2** (`docs/person2_ablation_p2_only.md`): P2 fusion alone, no dropout.
- **Person 3** (`docs/person3_ablation_dropout_only.md`): dropout alone, no P2.

Both configs are already written (`qfdet_r50_fpn_p2_only_vtuav.py`,
`qfdet_r50_fpn_dropout_only_vtuav.py`); Person 2's is smoke-tested, Person 3's is not yet —
their doc says so explicitly and asks them to verify before a full run.

**Once all three sets of numbers are in `results/metrics.csv`**, write
`reports/phase3_experimental_results.md`: headline comparison is fusion_v1's mAP_S vs.
baseline QFDet's mAP_S (0.129 test, from Phase 2), plus the ablation breakdown — does P2
alone explain most of the mAP_S gain, does dropout alone close the RGB-only-collapse gap
(Person 3's doc asks them to specifically re-run RGB-only/thermal-only eval on their
checkpoint to check this), or do the two changes need each other. Report the FLOPs/FPS cost
honestly alongside the accuracy numbers — same rigor as `reports/phase2_unimodal_baseline.md`.

Also still needed for Phase 3's deliverables (not done yet):
- [ ] Fusion architecture diagram (`reports/figures/phase3_fusion_architecture.png`)
- [ ] Strategy write-up (`reports/phase3_fusion_strategy.md`)
- [ ] Dashboard update (`reports/explorer/`) once real numbers exist, matching the Phase 2
  Model Results tab pattern

## Notes
- The earlier torchvision-based unimodal baselines were deleted (didn't use the provided
  pretrained weights, didn't satisfy the rubric — see `docs/phase2_unimodal_baseline.md`).
- All three ablation runs fine-tune from the same `checkpoints/qfdet_vtuav_pretrained.pth` —
  never from scratch, per the rubric.
