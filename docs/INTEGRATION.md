# Integration Guide — How the 4 Workstreams Combine

Each person's doc (`person1..4`) is written so they can work without touching each other's files. This doc explains the seams: where outputs land, what format they must be in, and why it's split this way. Read this if you're the one stitching results together (Person 4 / whoever assembles the final report).

## Why split this way (design rationale)

**By pipeline stage, not by person-as-generalist.** Phases 1/2/3/4 have a natural dependency order (dataset → unimodal → fusion → comparison), but Phase 2's two halves (RGB-only, thermal-only) and Phase 1 have *no* dependency on each other or on Phase 3 starting first. Splitting along those natural seams means 3 of the 4 people can start immediately in parallel instead of queuing behind the fusion model — that's the whole reason the 3-hour Phase 1+2 checkpoint is achievable at all.

**By compute constraint, not by arbitrary assignment.** Person 1 has the only 8GB card, so they own the one job that actually needs it (fusion training, which fine-tunes the full dual-branch QFDet). Persons 2/3's unimodal models are architecturally lighter — single input branch instead of two — so they fit on 6GB easily. Person 4 has no GPU, so they own the only phase with zero GPU dependency (dataset stats, visualization, writing). Nobody is stuck waiting on hardware they don't have.

**One shared contract, not shared code.** Rather than having everyone work inside one codebase live (merge conflicts, environment drift across 3 different machines), each person works in their own clone of the same repo + config pattern, and hands off only two things: a **checkpoint file** and a **metrics dict/table**. That's the entire integration surface. It's why the sections below only talk about file formats and naming, not shared modules.

## The integration surface

### 1. Checkpoints (Phase 2 + 3 → Phase 4)
Every trained model produces one `.pth` checkpoint file. Naming convention (already used in each person's doc) is what makes automatic aggregation possible without guessing what a file is:

| File | Produced by | Consumed by |
|---|---|---|
| `checkpoints/qfdet_vtuav_pretrained.pth` | Downloaded (see README.md) | Person 2/3's RGB-only/Thermal-only ablation eval, Person 1's Phase 3 fine-tuning starting point |
| `exp_fusion_v1.pth` | Person 1 | Phase 4 eval — this is the headline model |
| `exp_fusion_ablation_noattn.pth` | Person 1 (optional) | same |

**Update (Phase 2 complete):** RGB-only and Thermal-only are not separate checkpoints — the reference `QFDet` architecture has no single-modality mode, so both are evaluated from the one pretrained checkpoint above with a `ZeroModality` pipeline transform ablating one input stream. See `reports/phase2_unimodal_baseline.md` for the full methodology and results (superseded the original plan of Person 2/3 each training a separate torchvision model — see `STATUS.md` for why).

**Why a naming convention instead of a manifest file or database:** at hackathon speed, a naming convention that's grep-able is more robust than a coordination file that can go stale. Nobody has to remember to update a registry — the filename *is* the registry.

Push location: shared GitHub repo or shared Drive folder (decide once, put it in `STATUS.md` blockers section — currently unresolved, flagged there already).

### 2. Metrics (Phase 2/3/4 → Phase 5 report)
Each training/eval run produces the same fixed metric set — this is deliberate, not incidental: it's exactly what `FRD.md` Phase 4 needs for the master comparison table, so nobody computes a bespoke metric set that has to be reconciled later.

Required fields, always in this order, always this naming (matches mmdet's own COCO eval output keys so no manual renaming is needed):
```
mAP, mAP_50, mAP_75, mAP_s, mAP_m, mAP_l, FPS, inference_time_ms, model_size_MB, params_M, FLOPs_G
```
Drop these into a single shared `results/metrics.csv` (one row per experiment, columns = the fields above + an `experiment` name column matching the checkpoint naming). **Why CSV over each person writing prose findings:** Phase 4's comparison table and Phase 5's report both need to sort/diff numbers across 4-5 experiments — a flat table is the only format that makes "highlight the mAP_S delta" (the core differentiation argument from our strategy discussion) a one-line pandas filter instead of manual copy-paste across 4 people's writeups.

### 3. Predictions (Phase 3/4 → submission)
COCO JSON prediction files, one per (model × split) — really only val/test matter for submission, per the rules. Naming: `<experiment>_<split>_preds.json`, e.g. `exp_fusion_v1_test_preds.json`. This is literally the submission format the problem statement mandates, so there's no translation step at packaging time — whatever Phase 4 produces IS the deliverable.

### 4. Prose (Phase 1 + narrative → Phase 5 report)
Person 4's `reports/phase1_dataset_analysis.md` and any per-phase writeups drop straight into the final report's matching section (`FRD.md` Phase 5 lists the 1:1 section mapping already). No reformatting needed if everyone writes directly into report-shaped markdown from the start, rather than writing informal notes that need to be "translated" into report prose under time pressure later.

## What ties it all together at the end
`results/metrics.csv` + the 4-5 COCO JSON prediction files + `reports/phase1_dataset_analysis.md` + whatever narrative Person 1 writes about the fusion design choice = everything Phase 5 needs. Person 4 (or whoever is free first) assembles these into the final 3-5 page report and slides — they are not writing new analysis at that point, only arranging what already exists in report-ready form.

## Open item
Shared storage location (GitHub repo vs Drive) for checkpoints/predictions/metrics.csv is still not decided — see `STATUS.md` blockers. Pick one now so Phase 4 hand-offs aren't blocked later.
