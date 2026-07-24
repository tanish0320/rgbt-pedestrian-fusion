# Person 3 — Ablation: Modality Dropout Only (RTX 3050, 6GB)

Part of Phase 3's optional ablation study (`docs/phase3_fusion_plan.md` item 6): isolate
what modality-dropout training contributes on its own, without the P2/stride-4 fusion
change. Runs alongside Person 1's full `fusion_v1` (P2 + dropout, on the 4060) and Person
2's `p2_only` (P2, no dropout, on the other 3050).

## Setup
Pull this repo, then follow `README.md`'s Setup section (conda env, dataset, pretrained
checkpoint) exactly as written — nothing different for this machine.

## Your config
`third_party/mmdet-rgbtdroneperson/qfdet_configs/qfdet_r50_fpn_dropout_only_vtuav.py` —
already written, **not yet smoke-tested** (unlike Person 1's and Person 2's configs — that
test got interrupted before this handoff). Run the smoke test below yourself and treat any
unexpected error as a real finding to report back, not something to silently work around.

What it does: keeps the baseline FPN/anchor structure (no P2 — `start_level=1`,
`strides=[8,16,32,64,128]`, same as `qfdet_r50_fpn_1x_vtuav.py`, the original baseline), and
adds `RandomMasking` to `train_pipeline` (randomly zeroes RGB, zeroes thermal, or leaves both
real each training step — targets Phase 2's finding that RGB-only ablation collapsed 7×
while thermal-only only degraded 29%, evidence of an undertrained fusion gate). Fine-tunes
from the same pretrained checkpoint, same optimizer/epoch settings as the other two runs for
a fair comparison.

**Because there's no P2 here, this config's `nms_pre` is `1000` (baseline value), not `2000`
like the other two** — anchor count is unchanged from the original baseline, so no increase
was needed. This is intentional, don't "fix" it to match the other configs.

## Before training: run the smoke test
This config hasn't been verified end-to-end yet — you're the first real run of it.
```powershell
cd third_party/mmdet-rgbtdroneperson
$env:PYTHONPATH = (Get-Location).Path
python tools/smoke_test_p2.py --config qfdet_configs/qfdet_r50_fpn_dropout_only_vtuav.py --steps 5
```
**What to expect** (based on reasoning, not a confirmed prior run — verify this yourself):
- Checkpoint-load warnings for `atss_cls`/`bbox_prehead.atss_cls` only (num_classes 1 vs. 3
  mismatch, same reason as the other two configs) — but **no** `lateral_convs` mismatches
  this time, since the neck structure matches the checkpoint's original training config
  exactly (no P2 added here). If you *do* see lateral_conv mismatches, that's unexpected —
  stop and report it rather than proceeding, something would be wrong with the config.
- Peak VRAM should be lower than Person 2's P2 run (no stride-4 feature maps to hold in
  memory) — plausibly close to or below the original Phase 2 eval footprint. Not measured
  yet; report your actual number.
- If `CUDA out of memory` regardless, drop `data.samples_per_gpu` from `2` to `1` and re-run.

If the smoke test fails for any reason other than an expected OOM, don't debug it silently —
note the exact error and flag it, since this is genuinely the first execution of this config.

## Full training run
Only after the smoke test passes cleanly:
```powershell
python tools/train.py qfdet_configs/qfdet_r50_fpn_dropout_only_vtuav.py
```
6 epochs (matches the other two runs — don't change unless all three change together).
Checkpoint saves to `checkpoints/train_qfdet_dropout_only/` per-epoch, `work_dir` already
set in the config.

## Evaluation
Same pattern as Person 2's doc and `docs/phase2_unimodal_baseline.md`'s "Reproducing"
section, pointed at your checkpoint:
```powershell
cd third_party/mmdet-rgbtdroneperson
python tools/test.py qfdet_configs/qfdet_r50_fpn_dropout_only_vtuav.py checkpoints/train_qfdet_dropout_only/latest.pth --eval bbox
```
For test-split numbers, edit `data.test` to point at `test.json`/`mmdet_data/test/` the same
way Phase 2's `_test.py` configs do.

This run is also the one most directly comparable to Phase 2's RGB-only/thermal-only
ablation rows (`exp_rgb_only_qfdet_*`, `exp_thermal_only_qfdet_*` in `results/metrics.csv`)
— once you have real numbers, it's worth also running your fine-tuned checkpoint through the
`ZeroModality` RGB-only and thermal-only eval configs (same pattern, swap the checkpoint
path) to see whether modality-dropout training actually closed the 7×/29% gap from Phase 2.
That comparison is arguably the single most important number this ablation produces — don't
skip it even if time is short.

Also compute FPS/inference-time (`tools/benchmark_simple.py`) and FLOPs/params if time
allows, same as Phase 2's method.

## Report back
Add results as new rows to `results/metrics.csv`: `exp_dropout_only_val`,
`exp_dropout_only_test`, and if you ran the RGB-only/thermal-only re-eval,
`exp_dropout_only_rgbonly_test` / `exp_dropout_only_thermalonly_test` (or similar — keep the
naming consistent with existing rows).

Push your branch / open a PR rather than pushing straight to `main` — Person 1 will fold all
three ablation results into `reports/phase3_experimental_results.md` once all three are in.
