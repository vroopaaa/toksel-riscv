# In-Depth Hardware, Kernel, & Memory Stress Report (Method 2 Evaluation)

- **Device Tested**: cuda (NVIDIA GeForce GTX 1650)
- **ViT Architecture**: vit_small_patch16_224 (D=384, 12 layers)

## 1. Executive Kernel & Overhead Comparison

| Metric | Normal Dense ViT (1024) | Original Sparse-ViT (144) | Method 2 Max-Pool (144) | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **CUDA Kernel Count** | `574` | `4069` | **`1321`** | **Method 2 cuts 2748 kernels!** |
| **Sampler Latency** | `0.00 ms` | `12.31 ms` | **`0.49 ms`** | **25.0x faster sampler** |
| **Max Batch Size (VRAM)** | **B = 4** | **B = 12** | **B = 12** | Sparse fits **3.0x** larger batch |
| **Max FPS (Throughput)** | `20.8 FPS` | `144.8 FPS` | **`165.5 FPS`** | Peak throughput advantage |

## 2. Input Resolution Scaling Test (512x512 up to 1024x1024)

| Resolution | Dense Tokens | Sparse Tokens | Normal Dense ViT | Original Sparse-ViT | Method 2 (Max-Pool) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 512x512 | 1024 | 144 |  48.18 ms |  23.66 ms | ** 10.87 ms** |
| 640x640 | 1600 | 144 |  90.82 ms |  23.79 ms | ** 10.71 ms** |
| 768x768 | 2304 | 144 | 147.43 ms |  24.08 ms | ** 10.88 ms** |
| 896x896 | 3136 | 144 | 235.85 ms |  24.27 ms | ** 11.86 ms** |
| 1024x1024 | 4096 | 144 | 350.77 ms |  32.87 ms | ** 13.03 ms** |
