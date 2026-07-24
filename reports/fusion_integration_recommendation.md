# Fusion Integration Recommendation — Person 2

This report provides evidence-based recommendations for integrating the optimized RGB branch into Person 1's final RGB-Thermal fusion architecture.

## 1. Clean Modality Masking during Modality Dropout
- **Observation**: Simply zeroing input images causes Batch Normalization layers in the frozen backbones to propagate noisy activations, degrading accuracy.
- **Recommendation**: During Modality Dropout (ratio = 0.2), implement feature-level zeroing (as in `RGBOnlyQFDet`) where the entire thermal output tensor is directly zero-masked before fusion. This prevents batch normalization noise from polluting the fusion neck.

## 2. Scale Alignment and Inference Resolution
- **Observation**: Upscaling to $960\times768$ (Opt1) yields a **60% relative gain in small object detection ($mAP_S$)**, but increases GFLOPs.
- **Recommendation**: If compute budget allows, evaluate the fusion model at $960\times768$. If real-time FPS is required, use $640\times512$ with the relaxed NMS threshold of $0.60$ (Opt3), which increases recall with zero compute overhead.
