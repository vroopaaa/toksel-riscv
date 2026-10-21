# In-Depth Hardware, Kernel, & Memory Stress Report

- **Device Tested**: cuda (NVIDIA GeForce GTX 1650)
- **ViT Architecture**: vit_small_patch16_224 (D=384, 12 layers)

## 1. Executive Summary of Trade-offs & Downsides

| Performance Dimension | Normal Dense ViT (1024) | Proposed Sparse-ViT (144) | Verdict / Real-World Implication |
| :--- | :---: | :---: | :--- |
| **Transformer Latency** | `45.39 ms` | `5.58 ms` | **8.13x Speedup** inside ViT core! |
| **CUDA Kernel Count** | `574` launches | `4069` launches | **Downside**: +3495 extra kernels launched by sampler |
| **Sampler Bottleneck** | `0.00 ms` | `12.06 ms` | **Downside**: 47% of Sparse time spent in greedy NMS loop |
| **Max Batch Size (OOM)** | **B = 4** | **B = 12** | **Huge Win**: Fits **3.0x** larger batch in VRAM |
| **Peak Throughput** | `20.4 FPS` | `135.6 FPS` | **Huge Win**: Massive throughput scaling at higher batch sizes |

## 2. Micro-Dissection of the Fixation Sampler (The Bottleneck)

Why did the Fixation Sampler take `9.44 ms` in the first report? Here is the exact sub-millisecond breakdown:

| Internal Operation | Time (ms) | Percentage (%) |
| :--- | :---: | :---: |
| 1. Saliency/Objectness Min-Max & Fusion | 0.18 ms | 1.5% |
| 2. Coordinate Grid Calculation | 0.13 ms | 1.1% |
| 3. Pairwise cdist Distance Matrix (1024x1024) | 0.24 ms | 2.0% |
| 4. Sequential Greedy NMS Loop (128 iterations) | 11.36 ms | 94.3% |
| 5. Bilinear Continuous Score Sampling | 0.14 ms | 1.2% |
| **Total Fixation Sampler Time** | **12.06 ms** | **100.0%** |

## 3. Maximum Batch Size & OOM Stress Test

| Batch Size | Normal ViT Latency (ms) | Normal ViT FPS | Sparse-ViT Latency (ms) | Sparse-ViT FPS | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 56.06 ms | 17.8 | 26.13 ms | 38.3 | Both OK |
| 2 | 98.42 ms | 20.3 | 28.84 ms | 69.3 | Both OK |
| 4 | 192.77 ms | 20.7 | 41.10 ms | 97.3 | Both OK |
| 8 | **OOM (Crash)** | 0.0 | 61.56 ms | 130.0 | Sparse OK (Dense Crashed) |
| 12 | **OOM (Crash)** | 0.0 | 88.48 ms | 135.6 | Sparse OK (Dense Crashed) |

## 4. Input Resolution Scaling Test (512x512 up to 1024x1024)

| Resolution | Dense Tokens | Sparse Tokens | Dense Latency (ms) | Sparse Latency (ms) | Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 512x512 | 1024 | 144 | 48.23 ms | 22.68 ms | **2.13x** |
| 640x640 | 1600 | 144 | 92.11 ms | 22.54 ms | **4.09x** |
| 768x768 | 2304 | 144 | 147.51 ms | 23.10 ms | **6.38x** |
| 896x896 | 3136 | 144 | 230.90 ms | 23.59 ms | **9.79x** |
| 1024x1024 | 4096 | 144 | 353.53 ms | 24.08 ms | **14.68x** |
