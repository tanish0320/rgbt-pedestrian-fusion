# Phase 1 Deliverable Brief — for AI Coding Agents (Gemini, etc.)

This is a self-contained brief. You do not have access to prior conversation context — everything you need is below. If anything referenced here (a file, a path) doesn't exist when you check, say so explicitly rather than guessing.

## Context: what this is for

This is Phase 1 of a 4-phase hackathon project (24-hour hackathon: "Yugma TechFest 2.0 - MedhaDrishti National-Level AI Hackathon", challenge theme "AI for Multimodal RGB-Thermal Pedestrian Detection through Efficient Fusion Strategies"). The overall goal is to improve a baseline pedestrian-detection model (QFDet) that fuses RGB and thermal image pairs, with the biggest scoring weight (40%) going to a *novel fusion strategy* that specifically improves detection of **small/tiny pedestrians** — this is repeated three times in the problem statement as the hard, unsolved part.

Your job (Phase 1) does not touch the model at all. It is pure dataset analysis, and it runs in parallel with three other people/agents who are setting up training environments and running baseline models. **No GPU is needed or used for any of this.**

**Why Phase 1 matters beyond its own 15% score weight:** the analysis you produce here is deliberately meant to *set up the argument* for why the fusion model (built later, by someone else) should specifically target small pedestrians. Roughly 60 competing teams have the same dataset and will likely produce generic dataset stats. Your differentiation deliverables (section 2 below) exist to turn Phase 1 from a box-ticking exercise into actual evidence that gets reused in the final report's core pitch. Do not skip those sections even though they're extra work — they are the point.

## Where the data lives

Root: `C:\Users\urbra\OneDrive\Desktop\Projects\GG\VTUAV_subset\`

```
VTUAV_subset/
  annotations/
    train.json   -- COCO format, 1200 images, 8138 annotations
    val.json     -- COCO format, 300 images, 2337 annotations
    test.json    -- COCO format, 200 images, 2068 annotations
  VTUAV_co/{train,val,test}/images/   -- RGB ("visible"/"co") images, filenames like 00007.jpg
  VTUAV_ir/{train,val,test}/images/   -- thermal ("ir") images, SAME filenames as their RGB pair
  mmdet_data/{train,val,test}/{visible,thermal}/   -- Windows directory junctions pointing at the
                                                       VTUAV_co / VTUAV_ir folders above, used by the
                                                       training pipeline. You can read through either
                                                       path; they resolve to the same files.
```

Confirmed JSON structure (verified directly, not assumed):
```python
# categories
[{'id': 0, 'name': 'person'}]   # single class only

# one images[] entry
{'id': 6, 'file_name': '00007.jpg', 'height': 1080, 'width': 1920}

# one annotations[] entry
{'segmentation': [[801, 19, 801, 78, 842, 78, 842, 19]],
 'area': 2419, 'ignore': 0, 'iscrowd': 0, 'image_id': 6,
 'bbox': [801, 19, 41, 59],   # [x, y, width, height], COCO xywh format
 'category_id': 0, 'id': 18}
```
- `bbox` is `[x, y, w, h]` in pixel coordinates (top-left origin). `area` = `w * h` essentially (already precomputed, matches).
- To get the RGB/thermal pair for an annotation: look up `image_id` in `images[]` to get `file_name`, then that same filename exists in both `VTUAV_co/<split>/images/<file_name>` and `VTUAV_ir/<split>/images/<file_name>`.
- **Filenames are plain zero-padded numeric IDs with no encoded metadata** (no day/night flag, no scene tag). Do not assume you can parse lighting condition from the filename — confirmed by direct inspection. If a day/night breakdown is wanted, it can only come from visually inspecting images, not filename parsing.

## Official size-class thresholds (use these exact numbers, they are given by the competition rubric — do not invent your own bucket boundaries)
- **Small**: `area < 32²` (i.e. `area < 1024`)
- **Medium**: `32² <= area < 96²` (i.e. `1024 <= area < 9216`)
- **Large**: `area >= 96²` (i.e. `area >= 9216`)

## Deliverables

### Core (required)
1. **Dataset integrity check** — confirm image/annotation counts per split match what's stated above (1200/8138, 300/2337, 200/2068), confirm single category `person`. Report any discrepancy.
2. **Bbox size distribution histogram** — for each split, bucket every annotation's `area` into S/M/L using the thresholds above, plot a histogram (can be 3 subplots, one per split, or one grouped chart).
3. **Per-split S/M/L counts table** — rows: train/val/test, columns: count_small, count_medium, count_large, total. Plain numbers, e.g. a markdown table or CSV. This table gets reused later by a different phase (size-stratified mAP comparison), so keep the exact column names `count_small`/`count_medium`/`count_large`/`total`.
4. **Sample visualizations** — pick 3-5 image pairs (any split, prefer train), render: RGB image, thermal image, and an overlay or side-by-side, with all GT boxes for that image drawn on top of both. Save as PNG files.
5. **Alignment spot-check** — for 2-3 pairs, visually confirm the RGB and thermal images show the same scene/framing (i.e. a GT box drawn using train.json coordinates lands on the same real-world object in both the RGB and thermal image). Just state pass/fail with the example images as evidence — no need for a quantitative registration metric.
6. **(Optional, do only if time allows)** CLAHE (contrast-limited adaptive histogram equalization) on thermal images — try it on 2-3 samples, keep the before/after comparison if it visibly improves small-pedestrian visibility, otherwise skip and say so.

### Differentiation deliverables (do these — they are not optional busywork, see "why Phase 1 matters" above)
7. **Modality complementarity examples** — search for/curate 2-4 example image pairs where a pedestrian is hard to see in RGB (dark, low contrast, glare, blends into background) but clearly visible in thermal (heat signature stands out), and ideally 1 example of the reverse (thermal ambiguous, RGB clear — e.g. two people close together reading as one blob in thermal but distinct in RGB). Show both modalities side by side with the relevant pedestrian circled/boxed. This is the single most persuasive visual evidence for "why fusion is necessary" — treat it as the centerpiece figure of the whole Phase 1 report.
8. **Small-object visibility deep dive** — report the exact percentage of all annotations (pooled or per-split) that fall in the "small" bucket. Then pick 3-4 of the smallest annotated pedestrians in the dataset (lowest `area` values), crop a reasonably-padded region around each from both RGB and thermal, and display them zoomed in side by side. The point is to make visible, concretely, how little pixel signal a "small" pedestrian actually has — this previews the exact problem the project's fusion model is designed to solve.
9. **Per-split distribution shift check** — compare the S/M/L ratio (not raw counts — normalize by split size) across train/val/test. State plainly whether test/val skew towards more small objects than train (a real finding either way, don't force a conclusion that isn't there).
10. **Annotation density / occlusion proxy** — compute average pedestrians-per-image per split (`total_annotations / total_images`). Optionally identify the top 5 images by annotation count as candidate "crowd" scenes and note whether their boxes are smaller/more overlapping than average — small, clustered pedestrians in crowds are a plausible harder sub-case worth flagging.
11. **Day/night or scene-type breakdown** — since filenames carry no metadata (see note above), this can only be done by **visually sampling** a spread of images (e.g. every 100th image in train) and manually tagging apparent lighting condition (day / night / dusk-dawn / indoor-etc). If you do this, report it as an approximate/sampled estimate, not an exhaustive count. If time-constrained, skip this one and say so rather than fabricating a breakdown.

## Output format expected
Write everything into:
```
reports/phase1_dataset_analysis.md    -- the write-up: tables inline as markdown, images referenced by relative path
reports/figures/                       -- all PNG plots and sample visualizations referenced above
```
The markdown file should read as a stand-alone report section (it will be pasted close-to-verbatim into a larger final report), so use clear headers matching the deliverable numbers above, plain prose explaining what each figure/table shows, and avoid conversational filler ("Let's now look at...") — write it as report prose.

## Things to explicitly avoid
- Do not use any external/other dataset — only what's in `VTUAV_subset/`.
- Do not invent size thresholds different from the ones given above.
- Do not fabricate a day/night breakdown from filenames — they don't encode it (confirmed above).
- Do not touch or attempt to train/evaluate any model — that's out of scope for this phase entirely.
- If any expected file/folder is missing when you check, report that clearly instead of proceeding on assumptions.
