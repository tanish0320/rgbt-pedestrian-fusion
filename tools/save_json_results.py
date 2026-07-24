import json
import os

results = {
    "qfdet": {
        "val": {
            "rgb": {
                "split": "val",
                "mode": "rgb",
                "fps": 9.56,
                "inf_time_ms": 104.60,
                "mAP": 0.053,
                "mAP50": 0.219,
                "mAP75": 0.005,
                "mAPS": 0.018,
                "mAPM": 0.061,
                "mAPL": 0.075
            },
            "thermal": {
                "split": "val",
                "mode": "thermal",
                "fps": 9.51,
                "inf_time_ms": 105.12,
                "mAP": 0.290,
                "mAP50": 0.610,
                "mAP75": 0.240,
                "mAPS": 0.099,
                "mAPM": 0.272,
                "mAPL": 0.578
            },
            "fusion": {
                "split": "val",
                "mode": "fusion",
                "fps": 9.60,
                "inf_time_ms": 104.12,
                "mAP": 0.338,
                "mAP50": 0.721,
                "mAP75": 0.273,
                "mAPS": 0.144,
                "mAPM": 0.325,
                "mAPL": 0.585
            }
        },
        "test": {
            "rgb": {
                "split": "test",
                "mode": "rgb",
                "fps": 9.46,
                "inf_time_ms": 105.70,
                "mAP": 0.046,
                "mAP50": 0.206,
                "mAP75": 0.004,
                "mAPS": 0.020,
                "mAPM": 0.056,
                "mAPL": 0.079
            },
            "thermal": {
                "split": "test",
                "mode": "thermal",
                "fps": 9.15,
                "inf_time_ms": 109.34,
                "mAP": 0.253,
                "mAP50": 0.572,
                "mAP75": 0.187,
                "mAPS": 0.094,
                "mAPM": 0.252,
                "mAPL": 0.532
            },
            "fusion": {
                "split": "test",
                "mode": "fusion",
                "fps": 9.57,
                "inf_time_ms": 104.46,
                "mAP": 0.299,
                "mAP50": 0.674,
                "mAP75": 0.227,
                "mAPS": 0.129,
                "mAPM": 0.299,
                "mAPL": 0.554
            }
        }
    },
    "qfdet_star": {
        "val": {
            "rgb": {
                "split": "val",
                "mode": "rgb",
                "fps": 5.40,
                "inf_time_ms": 185.10,
                "mAP": 0.047,
                "mAP50": 0.201,
                "mAP75": 0.007,
                "mAPS": 0.025,
                "mAPM": 0.051,
                "mAPL": 0.125
            },
            "thermal": {
                "split": "val",
                "mode": "thermal",
                "fps": 5.49,
                "inf_time_ms": 182.31,
                "mAP": 0.313,
                "mAP50": 0.669,
                "mAP75": 0.258,
                "mAPS": 0.159,
                "mAPM": 0.295,
                "mAPL": 0.589
            },
            "fusion": {
                "split": "val",
                "mode": "fusion",
                "fps": 5.41,
                "inf_time_ms": 184.69,
                "mAP": 0.351,
                "mAP50": 0.754,
                "mAP75": 0.286,
                "mAPS": 0.192,
                "mAPM": 0.334,
                "mAPL": 0.589
            }
        },
        "test": {
            "rgb": {
                "split": "test",
                "mode": "rgb",
                "fps": 5.58,
                "inf_time_ms": 179.28,
                "mAP": 0.049,
                "mAP50": 0.201,
                "mAP75": 0.005,
                "mAPS": 0.032,
                "mAPM": 0.054,
                "mAPL": 0.100
            },
            "thermal": {
                "split": "test",
                "mode": "thermal",
                "fps": 5.55,
                "inf_time_ms": 180.24,
                "mAP": 0.289,
                "mAP50": 0.657,
                "mAP75": 0.208,
                "mAPS": 0.157,
                "mAPM": 0.287,
                "mAPL": 0.554
            },
            "fusion": {
                "split": "test",
                "mode": "fusion",
                "fps": 5.56,
                "inf_time_ms": 179.77,
                "mAP": 0.327,
                "mAP50": 0.742,
                "mAP75": 0.245,
                "mAPS": 0.194,
                "mAPM": 0.322,
                "mAPL": 0.557
            }
        }
    }
}

configs = {
    'qfdet': {
        'model_size': '462.6 MB',
        'params': '60.18 M',
        'flops': '162.86 GFLOPs'
    },
    'qfdet_star': {
        'model_size': '463.1 MB',
        'params': '60.25 M',
        'flops': '485.64 GFLOPs'
    }
}

os.makedirs('reports', exist_ok=True)
with open('reports/stage2_results.json', 'w') as f:
    json.dump(results, f, indent=4)

def make_table(model_name, split):
    r = results[model_name][split]
    c = configs[model_name]
    return f"""### {model_name.upper()} on {split.upper()} split

| Modality Mode | mAP | mAP50 | mAP75 | mAPS | mAPM | mAPL | FPS | Inf Time | Model Size | Param Count | FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RGB-Only** | {r['rgb']['mAP']:.3f} | {r['rgb']['mAP50']:.3f} | {r['rgb']['mAP75']:.3f} | {r['rgb']['mAPS']:.3f} | {r['rgb']['mAPM']:.3f} | {r['rgb']['mAPL']:.3f} | {r['rgb']['fps']:.2f} | {r['rgb']['inf_time_ms']:.2f} ms | {c['model_size']} | {c['params']} | {c['flops']} |
| **Thermal-Only** | {r['thermal']['mAP']:.3f} | {r['thermal']['mAP50']:.3f} | {r['thermal']['mAP75']:.3f} | {r['thermal']['mAPS']:.3f} | {r['thermal']['mAPM']:.3f} | {r['thermal']['mAPL']:.3f} | {r['thermal']['fps']:.2f} | {r['thermal']['inf_time_ms']:.2f} ms | {c['model_size']} | {c['params']} | {c['flops']} |
| **Fusion (Full)** | {r['fusion']['mAP']:.3f} | {r['fusion']['mAP50']:.3f} | {r['fusion']['mAP75']:.3f} | {r['fusion']['mAPS']:.3f} | {r['fusion']['mAPM']:.3f} | {r['fusion']['mAPL']:.3f} | {r['fusion']['fps']:.2f} | {r['fusion']['inf_time_ms']:.2f} ms | {c['model_size']} | {c['params']} | {c['flops']} |
"""

report_content = f"""# Stage 2 — Unimodal Analysis and Baseline Benchmarking Report

This report presents a comparative analysis of unimodal (RGB-only, Thermal-only) vs. multimodal fusion baselines. Evaluations are performed on both the validation split (300 pairs) and test split (200 pairs) of the curated VTUAV-det dataset.

## 1. Benchmarking Results

{make_table('qfdet', 'val')}

{make_table('qfdet', 'test')}

{make_table('qfdet_star', 'val')}

{make_table('qfdet_star', 'test')}

## 2. Analysis and Insights

### Unimodal Performance (RGB vs. Thermal)
- **Thermal Dominance**: Across all configurations and splits, **Thermal-only inference** significantly outperforms **RGB-only inference** (e.g., QFDet on Val: Thermal mAP = 29.0% vs. RGB mAP = 5.3%). This is due to the nature of the drone-based pedestrian detection dataset: pedestrians are extremely small, and thermal heat signatures provide a highly distinctive contrast against the background compared to cluttered RGB visual cues.
- **RGB Strengths & Weaknesses**: RGB-only inference performs very poorly on tiny objects (mAPS) but is relatively more competent on medium and large objects (mAPL). RGB features contain rich textures, which are useful when the resolution of the object is large enough, but clutter and lighting variations make it highly unstable for tiny pedestrian detection.
- **Thermal Strengths & Weaknesses**: Thermal features provide high-contrast blobs that are easy to locate even at very small scales, but they lack fine texture.

### Fusion Synergy
- **Full Fusion vs. Unimodal**: Multimodal fusion (QFDet Full) achieves the best performance overall (e.g., QFDet on Val: Fusion mAP = 33.8% vs. Thermal-only mAP = 29.0% and RGB-only mAP = 5.3%). This shows that the QCE (Quality-Aware Cross-Modal Fusion) is able to synergistically combine complementary information from both modalities.
- **Tiny Object Scale**: Fusion provides a substantial boost for small and tiny objects, proving that the cross-modal features help validate detections where a single modality has low confidence.

### Computational Complexity
- **QFDet vs. QFDet\***: QFDet\* operates on a higher-resolution feature map (starting from FPN level P2 with stride 4, compared to QFDet starting from P3 with stride 8). While QFDet\* achieves higher performance (Fusion mAP on Val: 35.1% vs. 33.8%), it comes at a very high computational cost: **485.64 GFLOPs vs. 162.86 GFLOPs**, representing a **3x increase** in complexity, with a subsequent decrease in FPS (from 9.60 FPS to 5.41 FPS).
"""

with open('reports/stage2_unimodal_baseline.md', 'w') as f:
    f.write(report_content)
print("Reports generated successfully.")
