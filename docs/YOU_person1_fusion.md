# Person 1 (You) — Fusion Lead — local RTX 4060

Your job is the 40%-weighted core stage. Everyone else's work feeds into or depends on yours, so get the baseline running first, then build the novel part.

## Everything runs locally — no Kaggle/Colab
This repo needs old pinned versions (torch 1.10.0, mmcv-full 1.6.1, mmdet 2.25.1).

conda env `qfdet` (torch 1.10.0+cu113) confirmed working with GPU detection (RTX 4060). `mmcv-full`/`mmdet` are **installed and verified** — `pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html` succeeded via the prebuilt wheel (network path to the CDN resolves now, no compiler needed), followed by `pip install mmdet==2.25.1`. Verified with a real `mmcv.ops.nms` call on the GPU.

**One dependency gotcha hit along the way:** `pip install opencv-python` on its own pulls numpy 2.x, which silently breaks torch 1.10.0's compiled C extensions (`_ARRAY_API not found` warning, not a hard failure, but not safe to ignore). Fixed by pinning `numpy<2` and `opencv-python<4.10`. If you rebuild this env from scratch, install opencv and numpy with those pins from the start rather than plain `pip install opencv-python`.

Config fixes still needed once mmdet imports cleanly: `num_classes=3→1`, dataset paths, ann_file names.

## Step 1 — Safety net (do this first, don't skip)
Fine-tune baseline QFDet from the pretrained checkpoint on the train set, then evaluate on val. This checkpoint is your fallback submission if the novel idea runs out of time. Update `STATUS.md` Phase 3 checkbox when done.

## Step 2 — The differentiator: small/tiny pedestrian fusion
The PDF calls out small/tiny pedestrian detection as the hard problem three separate times — this is what should make your fusion stand out from ~60 other teams.

**Idea:** add a cross-modal attention/gating fusion module at a shallow, high-resolution FPN level (P2/P3), not just deep layers. Rationale: small pedestrians lose signal after repeated downsampling, so fusing RGB+thermal early — before that loss happens — should specifically help mAP_S.

Where to look in the repo:
- `mmdet/models/detectors/` — find the `QFDet` detector class, see how it currently combines RGB/thermal features (`base_fusion='cat'` in the config — currently simple concatenation).
- `mmdet/models/necks/` (FPN) — this is where you'd hook in an earlier fusion point.
- Keep it to ONE clean module. Don't redesign the whole architecture — one well-motivated, well-ablated change beats three half-working ones.

Name checkpoints clearly: `exp_baseline_qfdet`, `exp_fusion_v1`.

## What to hand off to teammates
- Once `exp_baseline_qfdet` checkpoint exists → push to shared team storage, tell Teammate A/B — they'll run inference/eval on it in parallel on their 3050s (Phase 4).
- Once `exp_fusion_v1` checkpoint exists → same, plus flag it as "the headline model" for eval.
- Keep `STATUS.md` Phase 3 checkboxes current so Teammate C can pull real numbers into the report as they land.

## Local session tips
- **Save checkpoints periodically during long training**, not just at the end — a crash or reboot shouldn't cost you a full run.
- If interrupted mid-training, resume from the last saved epoch checkpoint (`--resume-from`) rather than restarting from scratch.
- The earlier torchvision-based unimodal baselines were deleted (they didn't use the provided pretrained weights, so didn't satisfy the rubric — see `docs/phase2_unimodal_baseline.md`). If a backbone-specialization ablation is wanted later, it would need to come from the real QFDet checkpoint's own `backbone`/`backbone_t` weights instead.
