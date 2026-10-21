# Hardware & Latency Profiling Report: Normal ViT vs. Sparse-ViT

- **Device**: cuda (NVIDIA GeForce GTX 1650)
- **Input Image Size**: 512 x 512
- **ViT Backbone**: vit_small_patch16_224

## 1. End-to-End Latency & FLOPs Comparison (Batch Size = 1)

| Model Architecture | Active Tokens | Params (M) | Total ViT FLOPs | Attention FLOPs | Latency (ms) | Throughput (FPS) | Speedup | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal Dense ViT (Baseline)** | 1024 | 22.14 M | 62.81 GFLOPs | 19.33 GFLOPs | 48.76 ± 0.18 ms | 20.5 FPS | **1.00x** | 375.9 MB |
| **Sparse-ViT (K=256)** | 272 | 22.69 M | 12.91 GFLOPs | 1.36 GFLOPs | 32.67 ± 1.33 ms | 30.6 FPS | **1.49x** | 365.9 MB |
| **Sparse-ViT (K=128)** | 144 | 22.69 M | 6.50 GFLOPs | 0.38 GFLOPs | 19.57 ± 0.18 ms | 51.1 FPS | **2.49x** | 365.9 MB |
| **Sparse-ViT (K=64)** | 80 | 22.69 M | 3.52 GFLOPs | 0.12 GFLOPs | 13.82 ± 0.18 ms | 72.4 FPS | **3.53x** | 365.9 MB |

## 2. Granular Stage-by-Stage Latency Breakdown Comparison

### Normal Dense ViT Stages (1024 Tokens)

| Stage | Latency (ms) | Percentage (%) |
| :--- | :---: | :---: |
| 1. Dense Patch Projection Conv2d | 0.57 ms | 1.2% |
| 2. Positional Embedding Interpolation | 2.28 ms | 4.7% |
| 3. Transformer Blocks (12 Dense Layers, 1024 tokens) | 45.39 ms | 92.8% |
| 4. Detection Head (Class & BBox for 1024 tokens) | 0.70 ms | 1.4% |
| **Total Normal ViT Pipeline** | **48.94 ms** | **100.0%** |

### Proposed Sparse-ViT Stages (144 Tokens)

| Stage | Latency (ms) | Percentage (%) |
| :--- | :---: | :---: |
| 1. CNN Feature Extractor (MobileNetV3) | 3.20 ms | 15.9% |
| 2. Fixation Sampler (Greedy NMS Loop) | 9.44 ms | 47.0% |
| 3. Local Patch Extractor (grid_sample) | 0.17 ms | 0.8% |
| 4. Patch Embedding & Pos/Score Injections | 1.47 ms | 7.3% |
| 5. Global Token Projection | 0.02 ms | 0.1% |
| 6. Transformer Blocks (12 Sparse Layers, 144 tokens) | 5.58 ms | 27.8% |
| 7. Detection Head (Class & BBox for 128 tokens) | 0.20 ms | 1.0% |
| **Total Sparse-ViT Pipeline** | **20.07 ms** | **100.0%** |

### Direct Stage Comparison Summary

- **Transformer Self-Attention Time**: Normal ViT = `45.39 ms` vs Sparse-ViT = `5.58 ms` (**8.13x faster**)
- **Sparse Fixation Preprocessing Tax**: `12.80 ms` (CNN + Sampler + grid_sample)
- **Net End-to-End Speedup**: **2.44x faster**

## 3. Batch Size Scaling (Throughput & Latency)

| Batch Size | Normal ViT Latency (ms) | Normal ViT FPS | Sparse-K128 Latency (ms) | Sparse-K128 FPS | Latency Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 48.95 ms | 20.4 | 19.76 ms | 50.6 | **2.48x** |
| 4 | 197.33 ms | 20.3 | 30.97 ms | 129.1 | **6.37x** |
| 8 | 392.45 ms | 20.4 | 56.59 ms | 141.4 | **6.94x** |
| 16 | 788.10 ms | 20.3 | 107.88 ms | 148.3 | **7.31x** |
