# Person 4 (No GPU) — Dataset Analysis (Phase 1, 15%) + Review Prep

**3-HOUR DEADLINE — review checkpoint.** You don't need a GPU for any of this — start immediately, in parallel with everyone else's environment setup.

## Your task: Phase 1 — Dataset Exploration & Analysis
Dataset already sitting at `VTUAV_subset/`:
- `annotations/{train,val,test}.json` — COCO format
- `VTUAV_co/{train,val,test}/images/` — RGB (visible) images
- `VTUAV_ir/{train,val,test}/images/` — thermal images
- Confirmed counts: train 1200 pairs (8138 boxes), val 300 pairs (2337 boxes), test 200 pairs (2068 boxes)
- Single category: `person`
- Size thresholds (use these exact numbers): Small = area < 32², Medium = 32² ≤ area < 96², Large = area ≥ 96² (px²)

### Concrete steps (Python, no GPU needed — just `json`, `matplotlib`, `PIL`/`opencv`)
1. **Bbox size distribution histogram** — load each split's JSON, compute `area` per annotation (already in the JSON as `bbox` width×height or the `area` field), bucket into S/M/L using the thresholds above, plot a histogram per split.
2. **Per-split S/M/L counts table** — a simple table: rows = train/val/test, columns = count of S/M/L boxes + total. This is a required number for the report and gets reused when Phase 4 breaks down mAP by size.
3. **Sample visualizations** — pick 3-5 image pairs, plot: RGB image, thermal image, and an overlay/side-by-side, with ground-truth boxes drawn on top (use `cv2.rectangle` or matplotlib patches from the `bbox` field in the JSON).
4. **Alignment spot-check** — visually confirm RGB and thermal images in a pair line up spatially (same scene, same box coordinates apply to both) — just eyeball 2-3 pairs.
5. **(Optional, skip if time-boxed)** CLAHE thermal contrast enhancement trial — only if the above 4 are done with time to spare.
6. Write up findings in `reports/phase1_dataset_analysis.md` — a few paragraphs + the histogram/table/sample images. This maps directly to Section 1 of the final report (see `FRD.md` Phase 5).

## Your task: Review Prep (due at 3-hour mark)
Since you're not blocked by GPU setup, you're the one pulling together what's ready for the review:
- Phase 1 findings (above) — should be complete or near-complete.
- Phase 2 numbers from Person 2 (RGB-only) and Person 3 (thermal-only) — pull in whatever they have, even partial/in-progress metrics.
- Phase 3 status from Person 1 — is baseline QFDet fusion running yet? What's the plan for the novel fusion piece?
- Put together a short status slide/doc for the review: what's done, what's in progress, what's next. Pull directly from `STATUS.md` — don't duplicate effort re-explaining, just summarize its current checkboxes.

## Note
Don't wait for perfect numbers — the review checkpoint is at 3 hours, Phase 2 GPU training may still be in progress. Report actual current state honestly (e.g. "RGB-only training at epoch 4/12, mAP so far X") rather than blocking on final numbers.
