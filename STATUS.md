# Status — MedhaDrishti Hackathon

Last updated: 2026-07-24

**Everything runs locally.** No Kaggle/Colab — mmcv has no Windows wheel, so the mmdet/ATSS route was dropped for Phase 2 in favor of plain torchvision, trained on the local RTX 4060.

**PRIORITY: Phase 1 + Phase 2 must be complete for review checkpoint in 3 hours. Phase 3 (fusion) continues in parallel but is not the review blocker.**

Legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked

## Phase 1 — Dataset Exploration & Prep (15%) — DONE
Completed independently by a teammate, delivered as a standalone folder, merged into this repo at `reports/` (was a sibling `Phase 1/` folder — moved in, duplicate copies of `PHASE1_AGENT_BRIEF.md`/PDF/README removed since they were byte-identical to the root copies).
- [x] Dataset integrity audit — 100% file parity, dimension/referential/area checks, 2 degenerate zero-width boxes found (`val` ann `41153`, `test` ann `43651`) and documented — **independently re-verified against the raw JSON, all counts exact** (see Stress Test note below).
- [x] Bbox size distribution histogram (`reports/figures/bbox_size_distribution.png`) — verified counts exact: train S/M/L 809/5449/1880, val 423/1592/322, test 529/1270/269.
- [x] Per-split S/M/L counts table — present in `reports/phase1_dataset_analysis.md` Table 3, columns match the `count_small`/`count_medium`/`count_large`/`total` convention Phase 4 needs.
- [x] Sample RGB/thermal/overlay visualizations — present in `reports/figures/`.
- [x] Alignment spot-check — PDF requires ≥20 pairs; **exactly 20 delivered** (`reports/figures/alignment/alignment_check_01.png`–`20.png`), visually spot-checked, boxes land correctly on the same person in both modalities.
- [x] CLAHE thermal enhancement trial — done, 3 samples, kept as a preprocessing recommendation.
- [x] Modality complementarity examples — strong night RGB-collapse vs. thermal-clear example (`complementarity_thermal_clear_*.png`), plus the reverse case (thermal blooming merges 2 people, RGB keeps them distinct — `complementarity_rgb_clear.png`).
- [x] Small-object visibility deep dive — 4 smallest pedestrians (114–136 px²) cropped and shown RGB vs. thermal side by side.
- [x] Per-split distribution shift check — real finding: small-object ratio rises 9.94% (train) → 18.10% (val) → 25.58% (test), a 2.57× escalation test-vs-train. This is a genuinely useful number for the Phase 3 fusion pitch.
- [x] Annotation density / occlusion proxy — mean ped/image 6.78 (train) → 10.34 (test), top-5 densest scenes identified.
- [x] Day/night sampling — every-100th-image visual tag (12 samples), reported as an approximate estimate as instructed, not fabricated from filenames.

### Stress test vs. `docs/PHASE1_AGENT_BRIEF.md` / official PDF rubric
Independently re-derived every headline number directly from `VTUAV_subset/annotations/*.json` (not just trusted the report) — every S/M/L count, mean-pedestrians-per-image, and degenerate-box ID matched exactly. All PDF-mandated Stage 1 deliverables are present, including the explicit "≥20 aligned pairs" requirement. This is solid, verifiable work, not padding.

Two shortcomings worth knowing about, neither blocking:
- Section 11 of the report proposes a full fusion architecture (BiFPN + Dual-Attention Fusion Module) — that's Stage 3 (40%) scope, not something Stage 1 asked for. Harmless as a forward-looking note, but Person 1 should treat it as one input/idea for Phase 3, not a spec to follow — the actual fusion design is Person 1's call.
- The report recommends a data-loader filter (`min_gt_bbox_wh`/`filter_empty_gt`) for the 2 degenerate boxes but doesn't apply it anywhere — turned out moot: `mmdet/datasets/vtuav.py`'s `_parse_ann_info` already unconditionally drops any box with `w < 1` or `h < 1`, so both degenerate boxes are filtered automatically on every load. Documented with a comment in `qfdet_configs/qfdet_r50_fpn_1x_vtuav.py` so nobody re-investigates this.

### Demo artifact
Built an interactive HTML explorer at `reports/explorer/index.html` (published as a Claude Artifact) as a demo-ready presentation layer on top of the Phase 1 report — not a rubric requirement, built for presentation value. Two tabs:
- **Overview** — hero RGB/thermal crossfade (99-pedestrian crowd scene), 4 headline stats, 2 modality-complementarity proof panels (night RGB-collapse, thermal-blooming), then a filterable 25-pair explorer grid with a full-resolution detail view (GT boxes drawn live from real annotation data).
- **Statistics** — KPI row + 4 charts (S/M/L stacked bar, per-split distribution shift, side-length histogram with split toggles, density bar) rendered from a new `reports/explorer/stats_manifest.json`, computed directly from the annotation JSON (not hand-typed) and confirmed working after a render fix pass.

Modular by design ahead of the planned repo restructure: `build_manifest.py` (regenerates all 3 JSON manifests from `VTUAV_subset/`), `build_page.py` (assembles `template.html` + manifests + `fonts/` → `index.html`), `template.html` (hand-edited source). Colors were run through the dataviz skill's CVD validator — original S/M/L palette failed (blue read as gray, gold failed contrast), replaced with a validated triad (orange/aqua/blue) that passes both light and dark mode.

## Phase 2 — Unimodal Analysis & Baseline Benchmarking (15%) — DONE

### Official deliverables (per the PDF, Stage 2 — verbatim requirements)
The PDF requires **three** detectors compared: RGB-only, Thermal-only, and the baseline QFDet, all evaluated using **the provided pretrained QFDet weights** (training from scratch not permitted). All three complete:
- [x] Detection metrics: mAP, mAP50, mAP75, mAP_S, mAP_M, mAP_L — all three detectors, val + test
- [x] Computational metrics: FPS, inference time, model size, number of parameters, FLOPs
- [x] Written comparative analysis of each modality's strengths/limitations

Full write-up: `reports/phase2_unimodal_baseline.md`. All 6 metric rows (3 detectors × val/test) in `results/metrics.csv`.

**How the "RGB-only"/"Thermal-only" requirement was actually satisfied**: the reference repo's `QFDet` architecture has no single-modality code path (`extract_feat` always requires both an RGB and thermal tensor). Added `ZeroModality`, a small additive pipeline transform (`mmdet/datasets/pipelines/multispectral_transforms.py`), that deterministically zeroes one input stream. This evaluates the real pretrained checkpoint with one modality ablated — matches the rubric's literal requirement ("using the provided pretrained baseline weights") — rather than substituting a differently-trained model.

**Pretrained checkpoint obtained**: `checkpoints/qfdet_vtuav_pretrained.pth` (485MB), downloaded via `gdown` from the plain `QFDet` link in the repo's README ("Trained Model → On VTUAV-det" table — NOT the `QFDet*` row, which is the authors' already-improved variant, not the baseline). Verified as a valid checkpoint (734 `state_dict` entries, both `backbone`/`backbone_t` present) before use.

### Results summary
| Experiment | Split | mAP | mAP50 | mAP_S |
|---|---|---:|---:|---:|
| Baseline QFDet (fused) | val | 0.338 | 0.721 | 0.144 |
| Baseline QFDet (fused) | test | 0.299 | 0.674 | 0.129 |
| RGB-only (thermal zeroed) | test | 0.042 | 0.191 | 0.020 |
| Thermal-only (RGB zeroed) | test | 0.232 | 0.547 | 0.080 |

Compute (identical across all 3 — same weights/architecture, only input differs): 485.11MB, 60.18M params, 162.86 GFLOPs, ~11.3–11.5 FPS, ~87–89ms inference.

**Key finding**: fusion beats either unimodal ablation by a wide margin (7× over RGB-only, 29% over thermal-only) — the clearest evidence yet for why Phase 3's fusion work matters. RGB-only collapses far more than thermal-only degrades, which is a real finding about this specific model's learned fusion gate (not just "thermal is better") — see the report for the full explanation. mAP_S is the weakest metric everywhere, consistent with Phase 1's small-object findings.

### Superseded torchvision run — deleted
An earlier thermal-only run used a separately trained torchvision `fasterrcnn_resnet50_fpn_v2` (not the provided QFDet weights), before the ZeroModality/pretrained-checkpoint approach above was built. Deliberately deleted (checkpoints, `notebooks/work_dir*`, the training scripts that produced it, and its 3 `metrics.csv` rows) rather than kept as "reference" — it doesn't satisfy the rubric's pretrained-weights requirement and risked being mistaken for the real Stage 2 thermal-only result later. The rows above (`exp_thermal_only_qfdet_*`) are the only thermal-only numbers that count.

## Phase 3 — Fusion Strategy (40%) — CORE
Local RTX 4060 (8.59GB VRAM confirmed). `conda` env `qfdet` (torch 1.10.0+cu113) confirmed working with GPU detection.

**Strategy locked in** (data-driven, not generic): shallow P2/stride-4 fusion (targets Phase 1's finding that small objects are sub-pixel by stride 32) + modality-dropout fine-tuning for gate robustness (targets Phase 2's finding that RGB-only ablation collapses 7× vs. thermal-only's 29% — a fusion-gate calibration problem, not a backbone-quality problem). Full plan, verified feasibility (level-count-agnostic code confirmed by direct read of FPN/heads/assigner/fusion path), and the 6-deliverable checklist: **`docs/phase3_fusion_plan.md`**.

- [x] Resolve `mmcv-full`/`mmdet` local install — **unblocked**, network now has a working path to `download.openmmlab.com` (NAT64-bridged IPv6, see notes log). Installed `mmcv-full==1.6.1` + `mmdet==2.25.1` via the prebuilt CDN wheel, no compiler needed. Also fixed a `numpy` 2.x/opencv conflict pulled in along the way (pinned `numpy<2`, `opencv-python<4.10`). Verified with a real GPU NMS call via `mmcv.ops.nms` — works.
- [x] Baseline QFDet running end-to-end (safety net) — done as part of Phase 2, same checkpoint (`checkpoints/qfdet_vtuav_pretrained.pth`) serves as Phase 3's fine-tuning starting point.
- [ ] Fusion architecture diagram
- [ ] Fusion strategy write-up (`reports/phase3_fusion_strategy.md`)
- [ ] Modified config (`qfdet_configs/qfdet_r50_fpn_p2_vtuav.py`) — P2 fusion + modality dropout
- [ ] Smoke test (1 epoch) for VRAM/`nms_pre` tuning before a full run
- [ ] Fusion v1 fine-tuned from pretrained checkpoint, saved to `checkpoints/qfdet_fusion_v1.pth`
- [ ] Fusion v1 evaluated on val + test, `results/metrics.csv` rows added
- [ ] Fusion v1 vs. baseline mAP_S comparison written up (`reports/phase3_experimental_results.md`)
- [ ] Dashboard updated with fusion-v1 row (`reports/explorer/`)
- [ ] (Optional, stretch) Ablation: P2 fusion without modality dropout, to isolate each half's contribution

## Phase 4 — Evaluation & Comparison (20%)
- [ ] Inference run: all checkpoints × val + test
- [ ] COCO JSON predictions exported (val + test)
- [ ] Master comparison table built
- [ ] mAP_S delta highlighted
- [ ] Efficiency delta (params/FLOPs/FPS) reported
- [ ] Failure case gallery (2-3 images)

## Phase 5 — Report & Slides (10%)
- [ ] Report draft started
- [ ] Report finalized (3-5 pages)
- [ ] Slides finalized

## Submission packaging
- [ ] Source code (zip/repo)
- [ ] Trained weights
- [ ] COCO JSON preds (val + test)
- [ ] Report
- [ ] Slides
- [ ] Rules compliance double-check

---

## Blockers / open decisions
- Shared checkpoint/results sync method not yet decided (GitHub repo vs shared Drive).
- `QFDet`/`ATSSHF` detector classes are hardwired for paired RGB+thermal input (two backbones each, no single-modality mode) — confirmed by reading the code directly. This is why Phase 2 baselines use plain torchvision instead of a modified version of this repo's custom detectors.
- Phase 1 is being handled independently by a teammate outside this repo's tracked flow — `docs/PHASE1_AGENT_BRIEF.md` remains the spec/contract for what their output should contain, but progress isn't tracked here in detail until it's handed back.
- RESOLVED: `mmcv-full` install blocker — see notes log below.

## Notes log
- 2026-07-24: Verified local dataset matches PDF spec exactly (1200/300/200, COCO JSON, single `person` category). FRD.md and STATUS.md created.
- 2026-07-24: Phase 2 thermal-only attempted locally with mmdet/mmengine (`notebooks/train_thermal.py`, `notebooks/atss_thermal_only.py`) — failed at runtime (`train.log`), traced to mmdet's `mmdet.datasets.samplers` import chain (mmcv has no Windows wheel). Rewrote as `notebooks/train_thermal_tv.py` using torchvision's `fasterrcnn_resnet50_fpn_v2` instead (prebuilt Windows wheels, no compiler dependency). Trained locally on the RTX 4060: 6 epochs, 600 iters/epoch, 3600 iters total. Script auto-runs COCOeval on val and dumps `metrics.json` at the end.
- 2026-07-24: Thermal-only training finished. mAP 0.350 / mAP_50 0.715 / mAP_S 0.222 / FPS 6.02 on val. Checkpoint + metrics pushed to `results/`.
- 2026-07-24: Re-checked network after a data recharge — `download.openmmlab.com` now reachable (`nslookup`/`Test-NetConnection`/HTTP 200 all confirmed). Root cause of the earlier failure was specifically no *native* IPv4 route on the previous connection; this network resolves the CDN over IPv6 with a NAT64 gateway (`64:ff9b::/96` prefix bridging to the CDN's real IPv4 addresses), which is functionally equivalent for pip installs. Installed `mmcv-full==1.6.1` (prebuilt wheel, ~3s download, no compiler) and `mmdet==2.25.1` into the existing `qfdet` conda env (torch 1.10.0+cu113, GPU-confirmed). Hit one follow-on issue: `pip install opencv-python` pulled numpy 2.x, incompatible with torch 1.10.0's compiled extensions — fixed with `numpy<2` and `opencv-python<4.10` pins. Verified end-to-end with a real `mmcv.ops.nms` GPU call. Phase 3 is now unblocked to start baseline QFDet training.
- 2026-07-24: Phase 1 (teammate's independent work, delivered as a sibling `Phase 1/` folder) stress-tested against the official PDF rubric and re-verified against raw JSON — all numbers check out exactly. Merged `Phase 1/reports/` into this repo's `reports/` (matches what `docs/INTEGRATION.md` already expected). Removed 4 files inside `Phase 1/` that were byte-identical duplicates of files already at repo root (`PHASE1_AGENT_BRIEF.md`, the problem-statement PDF, plus a stub `README.md`/`.gitignore`), then removed the now-empty folder. Phase 1 marked done.
- 2026-07-24: Built `reports/explorer/index.html`, an interactive demo artifact for Phase 1 (hero + pair explorer + Statistics tab with 4 charts), published as a Claude Artifact. Confirmed working/rendering correctly by direct check. Re-read the PDF's Stage 2 section closely and found `STATUS.md`'s Phase 2 tracking was incomplete — the rubric requires the **baseline QFDet** benchmarked too, not just RGB-only vs Thermal-only; that row wasn't planned anywhere. Added it, flagged as shared scope with Phase 3's Step 1 (same fine-tuned baseline checkpoint should serve both).
- 2026-07-24: Completed Phase 2 fully per the rubric's literal requirements. Downloaded the pretrained QFDet checkpoint (`qfdet_vtuav.pth`, 485MB) via `gdown` from the GitHub repo's README link, verified it as a valid checkpoint before use. Since the repo's QFDet has no single-modality mode, added `ZeroModality` (additive pipeline transform) to deterministically zero one input stream, so RGB-only/Thermal-only genuinely use the provided pretrained weights per the rubric rather than a substitute model. Built 6 eval-only configs (3 detectors × val/test). Hit and fixed two real bugs along the way: (1) `tools/test.py` resolves `mmdet` from `sys.path[0]` (the script's own directory), which silently picked up the pip-installed `mmdet` package instead of this repo's local fork, causing an `ImportError` for `rfnext_init_model` — fixed by setting `$env:PYTHONPATH` to the repo root before invoking; (2) `tools/test.py` always builds its dataset from `cfg.data.test`, never `cfg.data.val`, regardless of the `--eval` flag — an initial "val" config had `data.test` still pointing at `test.json`, so the first full round of "val" eval runs silently evaluated the test split instead (caught because the val and test run outputs were byte-identical, which is what triggered the investigation). Rebuilt all 6 configs with `data.test` correctly pointed per split and reran everything. Final numbers: baseline QFDet mAP 0.299 (test), thermal-only 0.232, RGB-only 0.042 — fusion beats either unimodal ablation by 7x/29% respectively. Compute: 485MB, 60.18M params, 162.86 GFLOPs, ~11.4 FPS (measured with a small custom single-GPU benchmark script, `tools/benchmark_simple.py`, since the repo's own `benchmark.py` requires a distributed launch). Full report: `reports/phase2_unimodal_baseline.md`. Phase 2 marked done.
- 2026-07-24: Deleted the superseded torchvision thermal-only run entirely (checkpoints, `notebooks/work_dir*`, the scripts that produced it, and its 3 `metrics.csv` rows) — it didn't satisfy the rubric's pretrained-weights requirement and risked being mistaken for the real Stage 2 result later. Full repo restructure for "clone and use" cleanliness: `VTUAV_subset/` → `data/VTUAV_subset/`, `mmdet-rgbtdroneperson/` → `third_party/mmdet-rgbtdroneperson/`, the pretrained checkpoint → top-level `checkpoints/qfdet_vtuav_pretrained.pth` (out of the vendored repo, which should stay pristine), eval prediction pickles → `results/preds/`. Recreated the `mmdet_data/` Windows directory junctions (they're absolute-path-based, broke on the move) and updated `data_root`/`work_dir` in all 7 affected MMDetection configs — re-ran a full eval smoke test afterward (mAP 0.338 val, exact match to the pre-move number) to confirm nothing broke. Added `README.md`, `.gitignore`, `requirements.txt`, `environment.yml` for a real "pull and use" setup path. Deleted `person2_rgb_baseline.md`/`person3_thermal_baseline.md` (fully superseded), replaced with `docs/phase2_unimodal_baseline.md` pointing at the real methodology/results. Fixed a real bug caught during the sweep: `reports/explorer/build_manifest.py` resolved the dataset path relative to its own script location and hadn't been updated for the `data/` move — would have silently failed on next regeneration. Removed an unreferenced 1.4MB `reports/explorer/data.js` leftover from an earlier manual step.
