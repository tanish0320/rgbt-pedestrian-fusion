# Functional Requirements Document (FRD)
## MedhaDrishti Hackathon — RGB-Thermal Pedestrian Detection (QFDet Fusion Improvement)

Source of truth: `Problem statement of AI for Object Recognition.pdf`. This FRD breaks the challenge into phases with concrete tasks, owners, and exit criteria. Update `STATUS.md` as tasks complete — this file describes *what*, status file tracks *progress*.

---

## Hard constraints (apply to every phase)
- No training from scratch — must fine-tune from provided pretrained QFDet weights.
- Only the provided VTUAV-det subset (1200 train / 300 val / 200 test pairs). No external data.
- Single class: `person` (category_id 0).
- Final predictions must be COCO JSON format, evaluated on the test set.
- Small/Medium/Large area thresholds: S < 32², 32² ≤ M < 96², L ≥ 96² (px²).
- Baseline repo: `third_party/mmdet-rgbtdroneperson` (MMDetection-based).

## Team & compute roster
| Person | Hardware | Primary role |
|---|---|---|
| You | Local RTX 4060, 8GB | Fusion model training (main trunk) |
| Teammate A | RTX 3050, 6GB | Unimodal baseline (RGB-only) + later: inference/eval runner |
| Teammate B | RTX 3050, 6GB | Unimodal baseline (Thermal-only) + later: ablation variant |
| Teammate C | No GPU | Dataset analysis, COCO JSON tooling, metrics aggregation, report/slides |

**Note:** everything runs locally, no Kaggle/Colab. `mmcv-full`/`mmdet` are installed and working (see `README.md`) — Phase 2's unimodal baselines and Phase 3's fusion model both use the real `QFDet` detector from `third_party/mmdet-rgbtdroneperson`, evaluated from the same provided pretrained checkpoint (`checkpoints/qfdet_vtuav_pretrained.pth`). See `reports/phase2_unimodal_baseline.md` for the unimodal methodology (a `ZeroModality` pipeline transform ablates one input stream rather than training a separate substitute model).

---

## Phase 1 — Dataset Exploration, Analysis & Preparation (15%) — DONE
**Owner:** Teammate C (no GPU needed).

All deliverables complete, written up in `reports/phase1_dataset_analysis.md`, figures in `reports/figures/`. Independently re-verified against the raw JSON (see `STATUS.md` notes log) — every headline number (instance counts, S/M/L buckets, degenerate boxes, density) matches ground truth exactly.

- [x] Dataset integrity: 1200/300/200 confirmed, 8138/2337/2068 annotations confirmed, category `person` only, 2 degenerate zero-width boxes found and documented.
- [x] Bbox size distribution histogram, official S/M/L thresholds.
- [x] Per-split S/M/L object counts table (`count_small`/`count_medium`/`count_large`/`total` columns, ready for Phase 4 reuse).
- [x] Sample RGB/thermal/overlay visualizations with GT boxes.
- [x] RGB/thermal alignment spot-check — PDF requires ≥20 pairs, exactly 20 delivered and verified.
- [x] CLAHE thermal contrast enhancement trial, 3 samples, kept as a recommendation.
- [x] Modality complementarity analysis — night RGB-collapse-vs-thermal-clear example, plus the reverse (thermal blooming merges two people, RGB keeps them distinct).
- [x] Small-object visibility deep dive — 4 smallest pedestrians (114-136 px²) cropped RGB vs thermal.
- [x] Per-split distribution shift check — real finding: small-object ratio 9.94% (train) → 25.58% (test), 2.57× escalation.
- [x] Annotation density / occlusion proxy — 6.78 (train) → 10.34 (test) ped/image, top-5 densest scenes flagged.
- [x] Day/night sampling (every-100th-image, 12 samples) — reported as an approximate estimate, not fabricated from filenames.

**Known non-blocking gaps:** the report's Section 11 proposes a full fusion architecture (BiFPN + attention module) — that's Stage 3 scope, treat as one input idea for Person 1, not a spec. The recommended data-loader filter for the 2 degenerate boxes isn't applied anywhere yet — a one-line addition when Phase 3's config is built.

**Exit criteria met.**

---

## Phase 2 — Unimodal Analysis & Baseline Benchmarking (15%)
**Owner:** Teammate A (RGB-only), Teammate B (Thermal-only) — both running locally (see `notebooks/train_thermal_tv.py` and `docs/person2_rgb_baseline.md`).

**Architecture note:** this repo's `QFDet`/`ATSSHF` detectors can't run single-modality (hardwired dual-backbone, checked directly in the code). MMDetection's stock `ATSS` was the original plan but `mmcv-full` has no prebuilt Windows wheel, so both baselines use plain **torchvision `fasterrcnn_resnet50_fpn_v2`** instead, fed one modality's images through a standard single-image COCO pipeline — kept architecturally identical between RGB-only and thermal-only except which image folder they read, so the comparison isolates modality, not architecture.

- [x] Thermal-only: fine-tuned + evaluated locally on the RTX 4060 (`notebooks/train_thermal_tv.py`, 6 epochs). val mAP 0.350, mAP_50 0.715, mAP_S 0.222, FPS 6.02.
- [ ] Fine-tune/evaluate RGB-only from COCO-pretrained backbone (same script, `VTUAV_co` images).
- [ ] For each: report full metric suite on val set —
  - Detection: mAP, mAP50, mAP75, mAP_S, mAP_M, mAP_L
  - Compute: FPS, inference time, model size, params, FLOPs
- [ ] **Export and keep the trained backbone weights specifically** (not just final checkpoints) — Phase 3 has an optional bonus ablation that reuses these as an alternate fusion-model initialization, so don't discard them after computing metrics.
- [ ] Save checkpoints + logs to shared repo/drive, naming `exp_rgb_only` / `exp_thermal_only` per `docs/INTEGRATION.md`.
- [ ] Write up: which modality wins where (expect thermal better at night/low-vis, RGB better at texture/detail; small objects likely weak in both — this motivates fusion).

**Exit criteria:** two-row comparison table (RGB-only vs Thermal-only) with full metrics, ready for Phase 4's combined table, with both trained backbone checkpoints preserved for Phase 3's optional bonus ablation.

---

## Phase 3 — Novel RGB-Thermal Fusion Strategy Development (40%) — CORE STAGE
**Owner:** You (RTX 4060), primary build. Teammate B assists with ablation variant once baseline fusion is stable.

- [ ] Step 1 (safety net): Get baseline QFDet fusion running end-to-end, fine-tuned on train set, evaluated on val. This is your fallback submission — do not skip or delay this.
- [ ] Step 2 (the differentiator): Implement ONE targeted architectural change aimed at small/tiny pedestrian detection:
  - Cross-modal attention/gating fusion module inserted at a shallow, high-resolution feature level (e.g. P2/P3 in FPN) rather than only deep layers.
  - Rationale to document: small pedestrians lose signal after repeated downsampling; fusing early preserves thermal heat-blob cues before that happens.
- [ ] Fine-tune this variant from the same pretrained weights.
- [ ] (Optional, if time allows) Ablation variant: same module without the attention gate, run in parallel, to isolate what the gate contributes.
- [ ] (Optional bonus, only after the above are safely done) **Backbone-specialization ablation**: using Phase 2's fully-trained RGB-only and thermal-only ATSS backbones, swap them in as the initialization for QFDet's `backbone`/`backbone_t` (instead of ImageNet weights) as a SEPARATE experiment, and compare against the required submission. This does not replace or risk the primary submission — the primary fusion checkpoint must still start from the provided pretrained QFDet weights per the hard constraints above. This bonus experiment exists only to strengthen the report with an extra data point ("does modality-specialized backbone initialization help fusion converge faster/better") — label it clearly as an ablation/experiment, not the submission, to stay compliant with the no-training-from-scratch rule.
- [ ] Track VRAM/compute usage; use AMP/fp16 + small batch size + gradient accumulation if needed.
- [ ] Save all checkpoints with clear naming: `exp_baseline_qfdet`, `exp_fusion_v1`, `exp_fusion_ablation_noattn`, `exp_fusion_ablation_specialized_backbone` (bonus, if attempted).

**Exit criteria:** at least one working improved-fusion checkpoint that beats baseline QFDet on val mAP_S, plus the baseline fusion checkpoint as fallback. Backbone-specialization ablation is bonus only — never blocks the required exit criteria.

---

## Phase 4 — Performance Evaluation & Comparative Analysis (20%)
**Owner:** Teammate A/B (whichever 3050 frees up first, inference-only so 6GB is fine) + Teammate C for aggregation.

- [ ] Run inference with all checkpoints (RGB-only, Thermal-only, baseline fusion, your fusion, ablation if present) on val AND test sets.
- [ ] Export predictions in COCO JSON format for val and test.
- [ ] Build the master comparison table: all models × all metrics (mAP/50/75/S/M/L, FPS, params, FLOPs).
- [ ] Highlight mAP_S delta specifically — this is the headline evidence for the differentiation story.
- [ ] Report efficiency delta: your fusion's added params/FLOPs/FPS cost vs baseline QFDet.
- [ ] Qualitative failure case gallery: 2-3 images where baseline misses a small pedestrian and your model catches it (and vice versa if honest reporting needs it).

**Exit criteria:** one master results table + qualitative gallery, ready to drop into report and slides.

---

## Phase 5 — Technical Report & Presentation (10%)
**Owner:** Teammate C drafts continuously as results land; you review technical accuracy.

- [ ] Report (3-5 pages), structured to mirror the rubric sections 1:1:
  1. Dataset exploration & prep
  2. Unimodal baseline results
  3. Fusion strategy — design + motivation (small-object focus)
  4. Evaluation & comparison (the master table + failure cases)
  5. Conclusion / efficiency note
- [ ] Slides: problem → unimodal numbers → fusion diagram → results table → small-object win → failure cases → efficiency note.

**Exit criteria:** report and slides finalized, all deliverables packaged (code, weights, COCO JSON preds for val+test, report, slides).

---

## Environment Setup (all machines)
- Python 3.10+, PyTorch, Torchvision, OpenCV, NumPy, MMDetection/OpenMMLab.
- Baseline repo: `third_party/mmdet-rgbtdroneperson` (clone: `https://github.com/NNNNerd/mmdet-rgbtdroneperson`).
- Dataset at `data/VTUAV_subset/` (VTUAV_co = RGB, VTUAV_ir = thermal, `annotations/{train,val,test}.json` in COCO format). Not tracked in git — see `README.md` for how to obtain it.
- Shared checkpoint/results sync: [decide — GitHub repo / shared Drive folder].

## Submission checklist (final)
- [ ] Source code (zip or GitHub repo)
- [ ] Trained model weights
- [ ] Predictions on val + test sets, COCO JSON format
- [ ] Technical report, 3-5 pages
- [ ] Presentation slides
- [ ] Sanity check: no external data used, no training-from-scratch, pedestrian-class only
