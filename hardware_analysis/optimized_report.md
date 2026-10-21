# Optimization Benchmark Report: Original vs. Optimized Samplers

- **Device**: cuda (NVIDIA GeForce GTX 1650)
- **Resolution**: 512x512
- **ViT Backbone**: vit_small_patch16_224

## 1. Standalone Fixation Sampler Latency Comparison

| Fixation Sampler Algorithm | Latency (ms) | Speedup vs Original | Description |
| :--- | :---: | :---: | :--- |
| **Original Brute-Force** | 8.62 ± 0.05 ms | 1.00x | 1024x1024 distance matrix + 128 loops |
| **Method 1: Direct 2D Grid Slicing** | 16.01 ± 0.57 ms | **0.54x** | Local 3x3 window zeroing on 2D grid |
| **Method 2: Max-Pool NMS (Zero-Loop)** | **0.37 ± 0.02 ms** | **23.20x** | Single parallel max-pool kernel (No loops) |

## 2. End-to-End Model Latency & Speedup (Batch Size = 1)

| Model Architecture | Active Tokens | End-to-End Latency (ms) | Throughput (FPS) | Speedup vs. Normal ViT |
| :--- | :---: | :---: | :---: | :---: |
| **Normal Dense ViT (Baseline)** | 1024 | 48.35 ± 0.12 ms | 20.7 FPS | **1.00x** |
| **Original Sparse-ViT (K=128)** | 144 | 19.29 ± 1.61 ms | 51.8 FPS | **2.51x** |
| **Sparse-ViT (Method 1: 2D Grid Slicing)** | 144 | 26.53 ± 0.24 ms | 37.7 FPS | **1.82x** |
| **Sparse-ViT (Method 2: Max-Pool NMS)** | 144 | 10.62 ± 0.06 ms | 94.2 FPS | **4.55x** |

## 3. Batch Size Scaling (Throughput & Latency)

| Batch Size | Normal ViT (ms / FPS) | Orig Sparse (ms / FPS) | Method 1 Grid Slicing (ms / FPS) | Method 2 Max-Pool (ms / FPS) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 48.6 ms (20.6) | 19.2 ms (52.2) | 26.6 ms (37.7) | **10.7 ms (93.7 FPS)** |
| 4 | 197.7 ms (20.2) | 33.9 ms (117.9) | 57.1 ms (70.1) | **24.4 ms (164.0 FPS)** |
| 8 | 393.5 ms (20.3) | 57.0 ms (140.4) | 98.0 ms (81.7) | **54.4 ms (147.0 FPS)** |
| 16 | 794.9 ms (20.1) | 108.2 ms (147.9) | 191.8 ms (83.4) | **108.9 ms (147.0 FPS)** |
