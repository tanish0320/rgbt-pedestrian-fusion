# Person 2 — Ablation: P2 Fusion Only (RTX 3050, 6GB)

Part of Phase 3's optional ablation study (`docs/phase3_fusion_plan.md` item 6): isolate
what the P2/stride-4 fusion level contributes on its own, without the modality-dropout
training change. Runs alongside Person 1's full `fusion_v1` (P2 + dropout, on the 4060) and
Person 3's `dropout_only` (dropout, no P2, on the other 3050).

## Setup
Pull this repo, then follow `README.md`'s Setup section (conda env, dataset, pretrained
checkpoint) exactly as written — nothing different for this machine. Confirm before training:
```powershell
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1e9)"
```
should show your RTX 3050 and ~6GB.

## Your config
`third_party/mmdet-rgbtdroneperson/qfdet_configs/qfdet_r50_fpn_p2_only_vtuav.py` — already
written and smoke-tested (5 real train steps on the 4060, no crashes, loss dropping
normally). You do not need to write or modify any config, only run it.

What it does: adds the P2/stride-4 FPN level (backbone already computes C2, the baseline
config just discarded it — see the config's own docstring for the full rationale, same as
Person 1's `fusion_v1` config). Fine-tunes from the same pretrained checkpoint. Does **not**
include `RandomMasking` (modality dropout) — that's the one deliberate difference from
`fusion_v1`, and the entire point of this run.

## VRAM — the one thing to verify on YOUR hardware
Smoke-tested on the 4060: peak VRAM 4,136–4,360 MB. Your 3050 has 6GB total (vs. the 4060's
8.59GB), so headroom is real but tighter — 4.1–4.4GB used out of 6GB should still fit, but
**run the smoke test first before committing to a full training run**:
```powershell
cd third_party/mmdet-rgbtdroneperson
$env:PYTHONPATH = (Get-Location).Path
python tools/smoke_test_p2.py --config qfdet_configs/qfdet_r50_fpn_p2_only_vtuav.py --steps 5
```
If you see a `CUDA out of memory` error, drop `data.samples_per_gpu` from `2` to `1` in the
config (halves batch size, roughly halves activation memory) and re-run the smoke test
before starting the full run. Report whatever value you end up using.

Expect to see checkpoint-load warnings on startup (3 `lateral_convs` + 2 `atss_cls` shape
mismatches) — this is expected and already documented in the config's docstring, not a bug.
Confirm the smoke test still ends with `SMOKE TEST PASSED` despite the warnings.

## Full training run
```powershell
python tools/train.py qfdet_configs/qfdet_r50_fpn_p2_only_vtuav.py
```
6 epochs (matches `fusion_v1` and `dropout_only` for a fair comparison — don't change this
unless all three runs change it together). Estimated ~50 min on the 4060's timing (0.8s/step,
600 iters/epoch); expect somewhat longer on the 3050 (slower GPU), budget more time.
Checkpoint saves to `checkpoints/train_qfdet_p2_only/` per-epoch, `work_dir` already set in
the config — no path changes needed.

## Evaluation
Once training finishes, evaluate on val and test using the same pattern as Phase 2 (see
`docs/phase2_unimodal_baseline.md`'s "Reproducing" section) — but pointed at your new
checkpoint instead of the pretrained one:
```powershell
cd third_party/mmdet-rgbtdroneperson
python tools/test.py qfdet_configs/qfdet_r50_fpn_p2_only_vtuav.py checkpoints/train_qfdet_p2_only/latest.pth --eval bbox
```
For test-split numbers, edit the config's `data.test` block to point at `test.json` /
`mmdet_data/test/` the same way the Phase 2 `_test.py` configs do (or ask if unsure — don't
guess the pattern).

Also compute FPS/inference-time (`tools/benchmark_simple.py`, same as Phase 2) and
FLOPs/params if time allows — same `get_model_complexity_info` wrapper pattern documented in
`reports/phase2_unimodal_baseline.md`'s Method section (note: P2 adds real FLOPs vs. the
Phase 2 baseline's 162.86G — expect a higher number here, that's expected, not a bug).

## Report back
Add your results as new rows to `results/metrics.csv`:
`exp_p2_only_val`, `exp_p2_only_test` — same 11-column schema as every other row in that file.

Push your branch / open a PR rather than pushing straight to `main` — Person 1 will fold all
three ablation results into `reports/phase3_experimental_results.md` once all three are in.
