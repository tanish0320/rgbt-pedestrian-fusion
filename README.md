# RGB-Thermal Pedestrian Detection — QFDet Fusion

Multimodal (RGB + thermal) pedestrian detection on the VTUAV-det dataset, built around the
[QFDet](https://github.com/NNNNerd/mmdet-rgbtdroneperson) baseline. Built for the Yugma
TechFest 2.0 "MedhaDrishti" AI Hackathon — see `Problem statement of AI for Object
Recognition.pdf` for the official rubric.

**Status**: Phase 1 (dataset analysis) and Phase 2 (unimodal baseline benchmarking) are
complete. Phase 3 (novel fusion strategy) is in progress. See `STATUS.md` for detailed,
continuously-updated progress and `FRD.md` for the full phase breakdown.

## Project structure

```
├── README.md                      this file
├── STATUS.md                      live progress tracker, read this first for "what's done"
├── FRD.md                         phase-by-phase requirements breakdown
├── docs/                          per-phase working notes and methodology
├── reports/                       write-ups and the interactive results dashboard
│   ├── phase1_dataset_analysis.md
│   ├── phase2_unimodal_baseline.md
│   └── explorer/                  interactive HTML dashboard (see below)
├── data/                          VTUAV_subset dataset — NOT in git, see Setup
│   └── VTUAV_subset/
├── checkpoints/                   model weights — NOT in git, see Setup
├── results/
│   ├── metrics.csv                tracked — every experiment's metrics, one row each
│   └── preds/                     COCO JSON / pickle predictions — NOT in git, regenerate with tools/test.py
└── third_party/
    └── mmdet-rgbtdroneperson/     vendored QFDet baseline repo (MMDetection fork)
        ├── qfdet_configs/         training + eval configs, including our added eval_*.py ones
        ├── mmdet/                 the framework; our one addition is ZeroModality
        │   └── datasets/pipelines/multispectral_transforms.py
        └── tools/
            └── benchmark_simple.py   our addition — single-GPU FPS benchmark (repo's own benchmark.py needs distributed launch)
```

## Setup

### 1. Clone and get the baseline repo
```bash
git clone <this-repo-url> GG
cd GG
```
`third_party/mmdet-rgbtdroneperson/` is already included as regular tracked files (a fork
with our additions — see below — not a submodule). If you're starting fresh instead of
cloning this repo, get it with:
```bash
git clone https://github.com/NNNNerd/mmdet-rgbtdroneperson third_party/mmdet-rgbtdroneperson
```

### 2. Python environment
Tested on Windows with Python 3.9, CUDA 11.3, an RTX 4060 (8GB). `mmcv-full`/`torch` are
version-pinned and order-sensitive — install in this order:

```bash
conda create -n qfdet python=3.9 -y
conda activate qfdet

# torch first, CUDA-version-specific — adjust cu113 if your GPU/driver needs a different build
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# mmcv-full needs a prebuilt wheel matching the exact torch+CUDA combo above
pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

# everything else
pip install -r requirements.txt
```

**Known gotcha**: installing `opencv-python` on its own can pull in NumPy 2.x, which
silently breaks torch 1.10.0's compiled extensions. `requirements.txt` already pins
`numpy<2` and `opencv-python<4.10` to avoid this — don't `pip install opencv-python` outside
of that file.

### 3. Dataset
Not tracked in git (634MB). Obtain `VTUAV_subset` from the organizers (see the PDF's
"Resources Provided by Organizers" section for the source) and place it at
`data/VTUAV_subset/`, matching this layout:
```
data/VTUAV_subset/
  annotations/{train,val,test}.json      COCO format
  VTUAV_co/{train,val,test}/images/      RGB images
  VTUAV_ir/{train,val,test}/images/      thermal images
  mmdet_data/{train,val,test}/{visible,thermal}/   directory junctions -> VTUAV_co/VTUAV_ir (Windows)
```
The `mmdet_data/` junctions are what the QFDet dataset loader actually reads. Recreate them
after placing the dataset (Windows):
```powershell
$root = "path\to\data\VTUAV_subset"
foreach ($split in @("train","val","test")) {
  New-Item -ItemType Junction -Path "$root\mmdet_data\$split\thermal" -Target "$root\VTUAV_ir\$split\images"
  New-Item -ItemType Junction -Path "$root\mmdet_data\$split\visible" -Target "$root\VTUAV_co\$split\images"
}
```
(On Linux/macOS, use symlinks instead of junctions — same target layout.)

### 4. Pretrained checkpoint
Not tracked in git (485MB). Download the baseline `QFDet` weights (trained on VTUAV-det,
**not** the `QFDet*` improved variant) from the link in
`third_party/mmdet-rgbtdroneperson/README.md`'s "Trained Model" table, and save it as
`checkpoints/qfdet_vtuav_pretrained.pth`:
```bash
pip install gdown
mkdir -p checkpoints
gdown "https://drive.google.com/uc?id=1Savf3oeiWek4eeXrvYLuaoBMoZW3nag8" -O checkpoints/qfdet_vtuav_pretrained.pth
```

## Running things

All commands assume the `qfdet` conda env is active and run from `third_party/mmdet-rgbtdroneperson/`
with `PYTHONPATH` pointed at that same directory (needed because `tools/test.py` otherwise
resolves the pip-installed `mmdet` package instead of this repo's local fork):

```bash
cd third_party/mmdet-rgbtdroneperson
export PYTHONPATH=$(pwd)   # PowerShell: $env:PYTHONPATH = (Get-Location).Path
```

**Evaluate the baseline / RGB-only / thermal-only detectors** (Phase 2 — see
`reports/phase2_unimodal_baseline.md` for full methodology):
```bash
python tools/test.py qfdet_configs/eval_qfdet_baseline_vtuav.py ../../checkpoints/qfdet_vtuav_pretrained.pth --eval bbox
python tools/test.py qfdet_configs/eval_qfdet_rgb_only_vtuav.py ../../checkpoints/qfdet_vtuav_pretrained.pth --eval bbox
python tools/test.py qfdet_configs/eval_qfdet_thermal_only_vtuav.py ../../checkpoints/qfdet_vtuav_pretrained.pth --eval bbox
```
Append `_test` to any config filename (e.g. `eval_qfdet_baseline_vtuav_test.py`) for
test-split numbers instead of val.

**Benchmark FPS / inference time**:
```bash
python tools/benchmark_simple.py qfdet_configs/eval_qfdet_baseline_vtuav.py ../../checkpoints/qfdet_vtuav_pretrained.pth
```

**Compute FLOPs / params** — the repo's `tools/analysis_tools/get_flops.py` doesn't handle
QFDet's paired `(rgb, thermal)` input by default; see the wrapper snippet in
`reports/phase2_unimodal_baseline.md`'s Method section.

**Regenerate the dataset-analysis dashboard** (`reports/explorer/index.html`):
```bash
python reports/explorer/build_manifest.py   # rebuilds JSON manifests from data/VTUAV_subset
python reports/explorer/build_page.py       # assembles the final self-contained HTML
```

## Results dashboard

`reports/explorer/index.html` is a self-contained interactive page (no server, no external
requests) covering:
- **Overview** — RGB/thermal complementarity examples, a filterable pair explorer with live-drawn ground-truth boxes
- **Statistics** — dataset distribution charts (size classes, density, per-split shift)
- **Model Results** — Phase 2's full metrics comparison table and written findings

Open it directly in a browser, or view it published at the Claude Artifact link (ask
whoever last published it, or republish with the Artifact tool pointed at this file).

## Key findings so far

- **Fusion beats either single modality by a wide margin**: Baseline QFDet (mAP 0.299,
  test) beats Thermal-only (0.232) by 29% and RGB-only (0.042) by 7×.
- **Small-object detection (mAP_S) is the weakest metric everywhere** — roughly a quarter
  of large-object mAP, in every configuration. This is the specific problem Phase 3's
  fusion strategy targets.
- **The dataset itself skews harder toward small objects at test time**: small-pedestrian
  share rises 2.57× from train (9.94%) to test (25.58%) — see
  `reports/phase1_dataset_analysis.md`.

Full detail in `reports/phase1_dataset_analysis.md` and `reports/phase2_unimodal_baseline.md`.
