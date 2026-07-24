# Phase 3 Implementation Plan — RGB-Thermal Fusion Strategy

Target: the 6 Stage 3 deliverables (fusion architecture diagram, description, modified
architecture, training methodology, experimental results, optional ablation), driven by
Phase 1/2's findings, not a generic fusion idea. This doc is the finishing checklist —
Phase 3 is done when every non-optional row below is checked and backed by a real artifact.

## Strategy recap (why this, not a generic attention module)

Two coupled changes to the existing `QFDet` architecture — not two separate models:

1. **Shallow-level (P2/stride-4) fusion** — targets Phase 1's finding that a 32×32px object
   (the small/medium boundary) collapses to a single feature cell at stride 32, and the
   smallest real annotations (~114px²) are already sub-pixel by then. QFDet's backbone
   already computes C2 (stride 4) — `out_indices=(0,1,2,3)` — but the FPN currently discards
   it (`start_level=1`). Extending the pyramid to include P2 is additive, not a rebuild.
2. **Modality-dropout fine-tuning for gate robustness** — targets Phase 2's finding that
   RGB-only ablation collapses 7× (mAP 0.042 vs. 0.299 fused) while thermal-only only
   degrades 29%. The existing `quality_attention` gate (`qce_fusion` in `qfdet.py`) was
   calibrated assuming both streams are reliable; it was never trained under the condition
   where one stream is weak/absent, which is common in this dataset (≈33% of sampled train
   images are night scenes, per Phase 1 §7).

Both changes touch the same fine-tuning run — this is one architecture + one training
recipe, evaluated once, not two independent experiments.

## Verified feasibility (checked directly in code before committing to this plan)

- FPN (`mmdet/models/necks/fpn.py`), both heads (`ATSSQHead`, `QFDetPreHead`), the fusion
  path (`qce_fusion`, `Fusion_strategy` in `qfdet.py`), and the assigner (`QLSAssigner`) are
  all **level-count-agnostic** — none hardcode 5 levels; all derive level count from
  `len(x_vs)` / `anchor_generator.strides` / `num_level_bboxes` at runtime. Confirmed by
  reading each file directly, not assumed from the config.
- Adding P2 needs **zero backbone changes** (C2 is already computed, just unused) and
  **zero head code changes** (config-only: extend `strides`, drop `start_level` to 0,
  `num_outs` 5→6).
- **Real cost, measured**: adding a stride-4 level roughly **quadruples total anchor
  positions** (6,820 → 27,300 at 640×512 input, +300%) because P2's feature map is 4× the
  area of P3's. This is the actual price of the small-object improvement — expect a real
  FLOPs/inference-time increase over the 162.86 GFLOPs / 88ms baseline, and it needs
  `test_cfg.nms_pre` (currently 1000) reconsidered given 4× more pre-NMS candidates.
- GPU: RTX 4060 Laptop, 8.59GB VRAM confirmed via `torch.cuda.get_device_properties`. P2's
  larger feature maps will raise activation memory during training — batch size may need to
  drop from the baseline config's `samples_per_gpu=2` if OOM occurs; not yet tested.

## Deliverables checklist

### 1. Fusion architecture diagram — not started
Plan: a single diagram showing the dual-backbone → 6-level FPN (P2–P7) → per-level
`qce_fusion` quality-gated fusion → shared `ATSSQHead`/`QFDetPreHead` path, with the new P2
tap and the modality-dropout training-time perturbation both called out visually against the
unmodified baseline. Build as a plain diagram (draw.io / a simple SVG/PNG), not prose —
export to `reports/figures/phase3_fusion_architecture.png`.

### 2. Description of the proposed fusion strategy — not started
Write-up covering: the two-part rationale above (grounded in Phase 1 §3.2's stride-32
feature-collapse math and Phase 2's RGB-only-collapse finding), why both changes are needed
together (P2 alone doesn't fix gate miscalibration; gate robustness alone doesn't fix
sub-pixel small objects), and what's explicitly *not* changed (backbone architecture,
`base_fusion='cat'` combination rule, loss functions) to keep the change auditable against
the baseline. Target: `reports/phase3_fusion_strategy.md`.

### 3. Modified network architecture — DONE
- [x] New config `qfdet_configs/qfdet_r50_fpn_p2_vtuav.py`: `neck.start_level=0`,
  `neck.num_outs=6`, both heads' `anchor_generator.strides` extended to
  `[4,8,16,32,64,128]`.
- [x] Modality dropout: found `RandomMasking` already existed in
  `mmdet/datasets/pipelines/multispectral_transforms.py` (unused by the baseline config,
  registered/exported already) — does exactly what was planned (random RGB-zero/thermal-zero/
  both-real per training step), so no new transform code was needed. Wired into
  `train_pipeline` at `p=(0.15, 0.15, 0.7)`, positioned before `MultiNormalize` (zeroing
  after normalization would produce a nonzero-mean tensor, not true black, since the
  configured means are ~83-135, not 0 — verified against `img_norm_cfg`).
- [x] `test_cfg.nms_pre` raised 1000→2000 pending real precision/recall data from a full eval.
- [x] Smoke-tested (5 real train steps, `tools/smoke_test_p2.py`): peak VRAM **4,446 MB** of
  8.59GB available (RTX 4060 Laptop) — comfortable headroom, no batch-size reduction needed.
  ~0.8s/step after warmup → ~8 min/epoch at 600 iters, so 6 epochs ≈ 50 min, feasible.
  Loss dropped 178.9→33.6 over 5 steps (reinitialized layers responding to gradient, healthy
  sign, not itself proof of final quality).
- [x] Checkpoint-load mismatch confirmed exactly as predicted: 3 lateral convs +
  `atss_cls`/`bbox_prehead.atss_cls` reinitialize (index-shift + num_classes reasons,
  documented in the config's own docstring), P2's new layers (`lateral_convs.3`,
  `fpn_convs.5`, `scales.5`) missing from checkpoint as expected — no surprises, no crash.

### 4. Training methodology — not started
- [ ] Fine-tune from `checkpoints/qfdet_vtuav_pretrained.pth` (never from scratch — required
  by the rubric).
- [ ] Document epoch count, LR schedule, batch size actually used (values TBD from the
  smoke test in item 3, not fixed in advance).
- [ ] Save the fine-tuned checkpoint as `checkpoints/qfdet_fusion_v1.pth`.

### 5. Experimental results — not started
- [ ] Eval on val and test using the same `tools/test.py --eval bbox` pipeline as Phase 2,
  for direct comparability.
- [ ] Compute FLOPs/params/FPS/inference-time the same way as Phase 2
  (`get_model_complexity_info` wrapper + `tools/benchmark_simple.py`).
- [ ] Append rows to `results/metrics.csv` (`exp_fusion_v1_val`, `exp_fusion_v1_test`).
- [ ] Headline comparison: fusion-v1 mAP_S vs. baseline QFDet mAP_S (0.129 test) — this is
  the number the whole strategy is built to move. Report the compute-cost delta alongside it
  (GFLOPs/FPS vs. the 162.86G/11.30fps baseline), since improving mAP_S at a real inference
  cost is a genuine tradeoff to report honestly, not hide.
- [ ] Write-up in `reports/phase3_experimental_results.md`, same rigor as
  `reports/phase2_unimodal_baseline.md` (verified numbers, written comparative analysis).
- [ ] Add a Model Results-style table to `reports/explorer/index.html`'s dashboard once
  numbers exist, matching the Phase 2 pattern already built there.

### 6. Ablation study — upgraded from stretch to active, using the 2 extra RTX 3050 laptops
Two isolation runs in parallel with Person 1's full `fusion_v1` run, splitting the strategy's
two changes apart to see what each contributes:
- [x] Configs written: `qfdet_r50_fpn_p2_only_vtuav.py` (P2 alone, no dropout),
  `qfdet_r50_fpn_dropout_only_vtuav.py` (dropout alone, no P2). Each differs from
  `qfdet_r50_fpn_p2_vtuav.py` by exactly one variable — verified via direct diff before
  handoff, not assumed.
- [x] `p2_only` smoke-tested on the 4060 (3 steps, clean checkpoint-load, no crash).
- [ ] `dropout_only` smoke test — not yet run before handoff, Person 3's doc asks them to
  verify it themselves and flag anything unexpected rather than debug silently.
- [ ] Both full training runs (Person 2, Person 3 — `docs/person2_ablation_p2_only.md`,
  `docs/person3_ablation_dropout_only.md`).
- [ ] Person 3's run should also re-evaluate RGB-only/thermal-only on their fine-tuned
  checkpoint (via the existing `ZeroModality` eval configs) — directly checks whether
  modality dropout closed Phase 2's 7×/29% collapse gap, arguably the most informative
  single number this ablation produces.

## Definition of done
Phase 3 is complete when: the diagram exists, the strategy write-up exists, the modified
config trains successfully end-to-end from the pretrained checkpoint, `results/metrics.csv`
has real `exp_fusion_v1_*` rows (not projected numbers), and the written comparison against
baseline QFDet's mAP_S is in `reports/phase3_experimental_results.md`. Ablation is bonus only.
