# Person 2 — Qualitative Analysis of the RGB Branch

This document analyzes the qualitative success and failure modes of our isolated RGB branch under varying UAV capture conditions.

## 1. Success Case Analysis
* **Daylight High-Contrast Scenes**: Tiny pedestrians are successfully detected down to $10\times10$ pixels. High-contrast visible details allow the model to recognize edge silhouettes and gait patterns.
* **Upscaled Scale Optimization (Opt1)**: Increasing resolution to $960\times768$ restores fine pedestrian forms that would otherwise be smoothed out by pooling layers. This results in clean detections of tiny objects at far distances.
* **Overlapping Pedestrians**: Relaxing the NMS threshold to $0.60$ preserves detections in overlapping crowd sequences.

## 2. Failure Case Analysis
* **Low Illumination / Night Operations**: In low light, the signal-to-noise ratio of the RGB sensor degrades. Pedestrians blend into shadows, leading to high false-negative rates. *(This is where the Thermal modality must compensate)*.
* **Low-Contrast Background Clutter**: Pedestrians wearing colors similar to roofs, asphalt, or dry grass are frequently missed by the RGB branch.
* **Occlusions**: Intermittent foliage or structure occlusions block visible appearance features, leading to partial detections or misclassifications.
