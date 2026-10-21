# Ultimate Analysis: Sparse-ViT-Det Hardware & Sampler Study

All hardware, latency and sampler analyses done in `hardware_analysis/` (29 Sep – 8 Oct 2026), in one place.

- **Hardware:** NVIDIA GeForce GTX 1650 (4 GB, laptop), PyTorch 2.11 + CUDA 13.0
- **Model:** `vit_small_patch16_224` backbone (D = 384, 12 layers, 6 heads), 512×512 input (32×32 grid) unless stated
- **Default token budget:** K = 128 fixation tokens + 16 global tokens = **144 tokens**, against **1024** for the dense ViT
- **Sources:** the saved JSON/MD/TeX results in this folder. Four print-only scripts that had never saved their output were re-run on 9 Oct 2026 and are marked **[re-run]**.

---

## 0. TL;DR

| # | Finding | Key number |
|---|---|---|
| 1 | Sparse tokens cut ViT compute about 10× (attention about 50×), but the original sparse pipeline only reached **2.5×** end-to-end, because of preprocessing overhead. | 62.81 → 6.50 GFLOPs; 48.76 → 19.57 ms |
| 2 | **The original greedy sampler was the bottleneck**: a Python loop of 128 steps on the GPU, about 4,000 profiler ops, 9–12 ms, **47–55 % of sparse latency**. The work itself is tiny; the cost is kernel-launch latency. | 94 % of sampler time is in the loop |
| 3 | **Method 2 (Max-Pool + Top-K)** makes the sampler about 20× faster (8.6 → 0.4 ms) and the whole pipeline about 1.9× faster at B = 1. | **10.6 ms, 94 FPS, 4.55× vs dense** |
| 4 | That advantage disappears at large batch sizes, because the transformer dominates and the CPU queues the loop's kernels ahead of the GPU. | B = 12: 81.1 vs 79.1 ms (1.02×) |
| 5 | Sparse latency is almost flat with resolution, while dense latency grows faster than the token count. | 1024²: 350.8 ms dense vs **13.0 ms** Method 2 (**26.9×**) |
| 6 | ⚠ **Method 2 does not select the same tokens as greedy on realistic maps.** It matches on noise maps (86 %), but on smooth saliency-like maps only about **26 %** of tokens overlap, and 89 % of Method 2 tokens sit next to another token. | Coverage 68.5 % (greedy) vs 47.2 % (Method 2) at K = 128 |
| 7 | **Exact greedy can be just as fast.** Sort-scan, bucket-scan and threshold+ variants on a uint8 map, run on the host, give the **same picks as greedy** and the **same end-to-end latency as Method 2** at B = 1. | 12.05 ms (sort-scan) vs 12.06 ms (Method 2) |
| 8 | Those variants need about **58× fewer memory operations** than the naive loop, which matters for the RISC-V target. | 1.05 M → 18.1 K loads+stores (K = 128, 64×64) |
| 9 | Bicubic positional interpolation is the largest embedding cost (about 1.2–1.6 ms). A GEMM-based **fast bicubic** gives the same result (≤ 1.4e-6 difference) and is 8–13× faster. Bilinear is also fast but changes the embeddings by about 9 % (relative L2). | 1.19 → 0.088 ms |
| 10 | The C ports of the greedy samplers match the golden picks on all 27 native tests. The rv32im/qemu leg crashes; one cause is found (see §11). | 27/27 native ✓ |

---

## 1. Pipeline and naming

```
Dense ViT : image → Conv2d patchify (1024 tok) → pos-embed interp → 12 ViT blocks → det head
Sparse-ViT: image → [1] MobileNetV3 CNN (saliency S, objectness O, 32×32)
                  → [2] Fixation sampler  F = 0.6·norm(S) + 0.4·norm(O) → K cells
                  → [3] Local patch extractor (grid_sample 16×16 patches)
                  → [4] Patch embed (4a) + pos-grid interp (4b) + Fourier pos MLP (4c)
                        + fixation-score MLP (4d) + 16 global tokens (4e)
                  → [5] 12 ViT blocks on 144 tokens → [6] det head
```

| Sampler | What it does | Same picks as greedy? |
|---|---|---|
| **Original greedy** (`models/fixation_sampler.py`) | 1024×1024 `cdist`, then K iterations of argmax and suppression of cells within `d_min` = 28 px (the 3×3 neighbourhood) | reference |
| **Method 1: grid slicing** (`optimized_samplers.py`) | Same loop, but zeroes a 3×3 window on the 2D grid and has no distance matrix | ✅ 100 % identical |
| **Method 2: Max-Pool NMS** (`optimized_samplers.py`, `backup_nms/`) | 3×3 `max_pool2d` peak mask; non-peaks scaled by 1e-4; one `topk` | ❌ only when the map has ≥ K peaks |
| **Host exact variants** (`greedy_variants.py`) | Quantise F to uint8 on the GPU, run the greedy on the host CPU: naive / rowmax / tilemax / sortscan / threshold / bucketscan / threshold+ | ✅ identical to the uint8 greedy |
| **C ports** (`greedy_c/`) | naive / sortscan / bucketscan for a 64×64 grid, written for RISC-V rv32im | ✅ native |

---

## 2. Index of analyses

| Date | Analysis | Script | Output |
|---|---|---|---|
| 29 Sep | Dense vs sparse: latency, FLOPs, K sweep, stages, batch | `profile_hardware.py` | `hardware_report.md`, `hardware_results.json` |
| 29 Sep | Kernel counts, sampler breakdown, OOM, resolution | `deep_kernel_profiler.py` | `deep_analysis_report.md/.json` |
| 29 Sep | Method 1 / Method 2 samplers, end-to-end, batch | `benchmark_optimized.py` | `optimized_report.md/.json` |
| 29 Sep | Kernel / OOM / resolution, now including Method 2 | `deep_kernel_profiler_method2.py` | `deep_analysis_method2_report.md/.json` |
| 29 Sep | Sampler fidelity (Method 1 and Method 2 vs greedy) | `verify_sampler_fidelity.py` | stdout only → **[re-run]** |
| 29 Sep | Stage breakdown across B = 1, 4, 8, 12 | `profile_large_batch_breakdown.py` | stdout only → taken from the TeX |
| 29 Sep | Combined report | — | `hardware_optimization_report.tex` |
| 1 Oct | In-model timings (`return_timings=True`), B = 1 and B = 12 | `test_native_timings.py` | `native_timings_results.json` |
| 1–6 Oct | Multi-image statistical profiling, stage-4 sub-stages | `profile_multi_image_analysis.py`, `profile_embedding_substages.py` | stdout (feeds later harnesses) |
| 6 Oct | Bicubic vs bilinear vs fast-bicubic positional interpolation | `profile_interp_mode_comparison.py`, `fast_bicubic.py` | `interp_mode_results.json` |
| 8 Oct | Greedy vs Max-Pool token overlap, noise vs smooth maps | `analyze_sampler_overlap.py` | stdout only → **[re-run]** |
| 8 Oct | 64×64 test vectors and golden picks for RISC-V | `export_sampler_testdata.py` | `sampler_testdata/` |
| 8 Oct | Exact greedy variants: correctness and memory-op counts | `greedy_variants.py` | stdout only → **[re-run]** |
| 8 Oct | Exact greedy variants inside the full pipeline (512 / 1024, B = 1 / 4) | `profile_greedy_variants.py` | `greedy_variants_pipeline.json` |
| 8 Oct | C samplers, native and rv32im (qemu) | `greedy_c/test_greedy_c.py` | stdout only → **[re-run]** |

---

## 3. Dense vs sparse baseline (original greedy sampler)

**512×512, B = 1** (`hardware_results.json`)

| Model | Tokens | Params | Total ViT GFLOPs | Attn GFLOPs | Latency (ms) | FPS | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense ViT | 1024 | 22.14 M | 62.81 | 19.33 | 48.76 ± 0.18 | 20.5 | 1.00× |
| Sparse K = 256 | 272 | 22.69 M | 12.91 | 1.36 | 32.67 ± 1.33 | 30.6 | 1.49× |
| Sparse K = 128 | 144 | 22.69 M | 6.50 | 0.38 | 19.57 ± 0.18 | 51.1 | 2.49× |
| Sparse K = 64 | 80 | 22.69 M | 3.52 | 0.12 | 13.82 ± 0.18 | 72.4 | 3.53× |

- The guidance CNN and samplers add only **+0.55 M parameters (+2.5 %)**.
- At K = 128, total FLOPs drop **9.7×** and attention FLOPs drop **50.6×**, but latency drops only **2.49×**. The gap is overhead, broken down in §4.

**Stage breakdown, B = 1**

| Dense stage | ms | % | | Sparse (K = 128) stage | ms | % |
|---|---:|---:|---|---|---:|---:|
| Patch Conv2d | 0.57 | 1.2 | | 1. CNN (MobileNetV3) | 3.20 | 15.9 |
| Pos-embed interp | 2.28 | 4.7 | | **2. Fixation sampler** | **9.44** | **47.0** |
| **12 ViT blocks (1024 tok)** | **45.39** | **92.8** | | 3. Patch extractor | 0.17 | 0.8 |
| Det head | 0.70 | 1.4 | | 4. Embed + pos/score | 1.47 | 7.3 |
| | | | | 5. Global token proj | 0.02 | 0.1 |
| | | | | 6. 12 ViT blocks (144 tok) | 5.58 | 27.8 |
| | | | | 7. Det head | 0.20 | 1.0 |
| **Total** | **48.94** | | | **Total** | **20.07** | |

- The transformer itself is **8.13× faster** (45.39 → 5.58 ms).
- Sparse preprocessing (CNN + sampler + grid_sample) costs **12.80 ms**, which is more than the whole sparse transformer.

**Batch scaling** (no_grad)

| B | Dense ms (FPS) | Sparse K = 128 ms (FPS) | Speedup |
|---:|---:|---:|---:|
| 1 | 48.95 (20.4) | 19.76 (50.6) | 2.48× |
| 4 | 197.33 (20.3) | 30.97 (129.1) | 6.37× |
| 8 | 392.45 (20.4) | 56.59 (141.4) | 6.94× |
| 16 | 788.10 (20.3) | 107.88 (148.3) | 7.31× |

The dense model is already compute-saturated at B = 1: its FPS stays flat at about 20.4. The sparse model's FPS keeps rising until about 148 FPS.

---

## 4. Why the original sampler was slow

**Inside the sampler** (`deep_analysis_results.json`)

| Operation | ms | % |
|---|---:|---:|
| Min-max normalisation and fusion | 0.18 | 1.5 |
| Coordinate grid | 0.13 | 1.1 |
| `cdist` 1024×1024 | 0.24 | 2.0 |
| **Sequential greedy loop (128 iterations)** | **11.36** | **94.3** |
| Bilinear score sampling | 0.14 | 1.2 |
| **Total** | **12.06** | |

**Profiler ops per forward pass** (`deep_analysis_method2_results.json`)

| | Dense | Original sparse | Method 2 |
|---|---:|---:|---:|
| Profiler kernel/op events | 574 | **4,069** | 1,321 |
| Profiler CPU time (ms) | 114.9 | 103.0 | 48.5 |
| Profiler CUDA time (ms) | 261.7 | 80.9 | 52.6 |

The most frequent ops in the original sampler are `reshape` ×303, `clamp` ×260 and `index` ×256, all per-iteration loop work. The loop is **bound by launch and dispatch latency, not by compute**. §10 supports this: even a naive numpy greedy on the CPU (1.48 ms, including transfers) is 6.8× faster than the GPU loop (10.04 ms).

> These counts are torch-profiler events (aten ops plus device kernels), not pure CUDA kernel launches. The ratio between the models is what matters.

---

## 5. Sampler optimisation round 1: Method 1 vs Method 2

**Sampler alone, B = 1** (`optimized_results.json`)

| Sampler | ms | vs original |
|---|---:|---:|
| Original (cdist + loop) | 8.62 ± 0.05 | 1.00× |
| Method 1: 2D grid slicing | 16.01 ± 0.57 | **0.54× (slower)** |
| **Method 2: Max-Pool + Top-K** | **0.37 ± 0.02** | **23.2×** |

Method 1 removes the distance matrix but keeps the loop, so it gets slower: each 2D slice launches more ops. This confirms that the loop, not the `cdist`, is the problem.

**End-to-end, B = 1**

| Model | ms | FPS | vs dense |
|---|---:|---:|---:|
| Dense ViT | 48.35 ± 0.12 | 20.7 | 1.00× |
| Original sparse | 19.29 ± 1.61 | 51.8 | 2.51× |
| Sparse + Method 1 | 26.53 ± 0.24 | 37.7 | 1.82× |
| **Sparse + Method 2** | **10.62 ± 0.06** | **94.2** | **4.55×** |

**K sweep with Method 2** (from the TeX only; no raw results were saved, see §13)

| K | Original ms | Method 2 ms | Method 2 FPS | Method 2 vs dense |
|---:|---:|---:|---:|---:|
| 256 | 32.67 | 15.34 | 65.2 | 3.18× |
| 128 | 19.57 | 10.62 | 94.2 | 4.55× |
| 64 | 13.82 | 7.45 | 134.2 | 6.54× |

---

## 6. Batch scaling: the Method 2 advantage shrinks

**Stage breakdown across batch sizes, K = 128** (from `profile_large_batch_breakdown.py`, as reported in the TeX)

| Stage (ms) | B=1 Orig | B=1 M2 | B=4 Orig | B=4 M2 | B=8 Orig | B=8 M2 | B=12 Orig | B=12 M2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. CNN | 2.91 | 2.80 | 4.76 | 4.90 | 9.20 | 9.31 | 13.68 | 13.74 |
| **2. Sampler** | **8.83** | **0.44** | **7.01** | **0.12** | **2.71** | **0.12** | **2.80** | **0.13** |
| 3. Patch extract | 0.16 | 0.14 | 0.19 | 0.15 | 0.30 | 0.30 | 0.44 | 0.44 |
| 4. Embed + pos/score | 1.75 | 1.53 | 2.13 | 2.23 | 2.76 | 2.81 | 3.56 | 3.55 |
| **5. ViT blocks** | 6.24 | 5.63 | 15.81 | 16.34 | 39.28 | 39.94 | **59.21** | **59.89** |
| 6. Det head | 0.22 | 0.20 | 0.52 | 0.55 | 0.86 | 0.86 | 1.40 | 1.41 |
| **Total** | **20.10** | **10.74** | **30.43** | **24.28** | **55.11** | **53.33** | **81.08** | **79.14** |
| FPS | 49.8 | 93.1 | 131.4 | 164.7 | 145.2 | 150.0 | 148.0 | 151.6 |
| M2 speedup | 1.87× | | 1.25× | | 1.03× | | 1.02× | |

**In-model timings** (`native_timings_results.json`, 30 runs; `return_timings=True`)

| | B=1 Orig | B=1 Method 2 | B=12 Orig | B=12 Method 2 |
|---|---:|---:|---:|---:|
| CNN | 4.30 ± 0.78 | 4.44 ± 0.90 | 13.93 | 13.93 |
| Sampler | **11.85 ± 1.88** | **0.75 ± 0.19** | 3.74 ± 2.49 | 0.13 |
| Patch extract | 0.29 | 0.22 | 0.47 | 0.44 |
| 4a patch embed | 0.27 | 0.20 | 1.16 | 1.16 |
| **4b positional encoding** | **1.40** | **1.37** | **2.07** | **2.10** |
| 4c score embed | 0.03 | 0.03 | 0.22 | 0.22 |
| 4d global tokens | 0.05 | 0.03 | 0.12 | 0.12 |
| ViT blocks | 6.31 | 5.65 | 59.98 | 60.69 |
| Det head | 0.20 | 0.20 | 1.42 | 1.43 |
| **Total** | **24.70 ± 2.67** | **12.89 ± 1.19** | **83.11 ± 2.25** | **80.22 ± 0.17** |
| FPS | 40.5 | **77.6** | 144.4 | 149.6 |
| Sampler / total speedup | 15.8× / **1.92×** | | 28.4× / **1.04×** | |

**Why the gap closes** (Amdahl's law plus asynchronous queueing):

1. The ViT blocks run the same B×144×384 tensor in both models (59.2 vs 59.9 ms at B = 12). At B = 12 they take more than 73 % of the pipeline.
2. At B ≥ 8 the CNN runs for 9–14 ms on the GPU. Meanwhile the CPU runs ahead and queues all of the greedy loop's kernels, so the original sampler drops from about 9 ms to **2.7 ms**.
3. Method 2 also has much lower variance: ±0.17 ms against ±2.25 ms total at B = 12, which helps a real-time worst-case bound.

---

## 7. Resolution scaling (B = 1, K = 128 fixed)

(`deep_analysis_method2_results.json`)

| Resolution | Dense tokens | Dense (ms) | Original sparse (ms) | Method 2 (ms) | Original vs dense | **Method 2 vs dense** |
|---|---:|---:|---:|---:|---:|---:|
| 512² | 1024 | 48.18 | 23.66 | 10.87 | 2.04× | **4.43×** |
| 640² | 1600 | 90.82 | 23.79 | 10.71 | 3.82× | **8.48×** |
| 768² | 2304 | 147.43 | 24.08 | 10.88 | 6.12× | **13.55×** |
| 896² | 3136 | 235.85 | 24.27 | 11.86 | 9.72× | **19.89×** |
| 1024² | 4096 | 350.77 | 32.87 | 13.03 | 10.67× | **26.92×** |

- Dense tokens grow 4× and dense latency grows 7.3×, showing the quadratic attention term.
- Sparse latency stays nearly flat. Only the CNN and the sampler grid grow with resolution.
- An earlier run (`deep_analysis_results.json`) agrees, with the original sampler at 2.13× → 14.68×.

---

## 8. Memory and OOM limits

(`deep_analysis_method2_results.json`)

| Model | Peak VRAM at B = 1 | B = 4 | B = 12 | Max B | OOM at |
|---|---:|---:|---:|---:|---:|
| Dense ViT | 895 MB | 2,704 MB | — | 4 | 8 |
| Original sparse | 453 MB | 1,011 MB | 2,479 MB | 12 | 16 |
| Method 2 sparse | 453 MB | 1,011 MB | 2,479 MB | 12 | 16 |

Method 2 peak throughput in this test: **165.5 FPS at B = 4**.

> ⚠ **Caveat:** `run_oom_stress_test()` runs **without `torch.no_grad()`**, so autograd activations are kept and these limits reflect a forward pass with gradients enabled. In the inference benchmarks, which do use `no_grad` (§3, §5), dense ran fine at **B = 8 and B = 16**. So the claim that sparse fits a "3× larger batch" holds only for grad-enabled passes. For inference it needs re-measuring under `no_grad`. Each OOM-test latency also comes from a single timed run.

---

## 9. Token fidelity: does Method 2 pick the same tokens as greedy?

There are three measurements, and they disagree.

**(a) Claimed in the TeX report.** No saved output backs these numbers, and no script computes saliency-mass retention.

| Metric | Value claimed |
|---|---|
| Top-1 peak match | 100 % |
| Within 1.5 patches | 92.4–95.3 % |
| Saliency mass retention | 98.8 % |
| Exact cell match | 68.5–74.2 % |

**(b) `verify_sampler_fidelity.py`** **[re-run]** (20 trials, i.i.d. random maps, K = 128)

| | Value |
|---|---|
| Method 1 vs greedy | **100 % identical**, same order, 0 px difference |
| Method 2 exact same cell | 86.2 % |
| Method 2 within 1 cell (≤ 16 px) | 87.0 % |
| Mean distance to the nearest greedy pick | 4.47 px |

**(c) `analyze_sampler_overlap.py`** **[re-run]** (200 maps per type)

| Map type | K | 3×3 peaks in F | Method 2 filler (non-peak) % | **Exact overlap %** | Greedy unique cells (min) | Adjacent tokens % G / MP | Coverage % G / MP |
|---|---:|---:|---:|---:|---:|---:|---:|
| i.i.d. noise | 64 | 121.0 | 0.0 | 97.3 | 64 (64) | 0.0 / 0.0 | 83.7 / 84.2 |
| | 128 | 121.0 | 5.7 | 85.7 | 128 (128) | 0.0 / 10.7 | 99.2 / 99.5 |
| | 256 | 121.0 | 52.7 | 62.3 | 200.7 (190) | 0.6 / 82.3 | 100 / 100 |
| **smooth saliency-like** (2–6 Gaussian blobs) | 64 | 15.8 | 75.3 | **21.2** | 64 (64) | 0.0 / 78.0 | 38.8 / 37.8 |
| | **128** | 15.8 | **87.6** | **25.9** | 128 (128) | **0.0 / 89.4** | **68.5 / 47.2** |
| | 256 | 15.8 | 93.8 | **29.4** | 241.0 (222) | 0.5 / 94.9 | 100 / 61.9 |

**What this means**

- Method 2 matches greedy only when F has at least K local maxima. Noise maps have about 121 peaks per 32×32 map, which is why (b) looks good.
- Smooth, realistic maps have only about **16 peaks**. Method 2 then fills the remaining 112 slots with the highest-scoring non-peak cells, which are the cells right next to the peaks. Its tokens **cluster**: 89 % sit next to another token, and grid coverage drops from 68.5 % to 47.2 %. In effect Method 2 is a plain Top-K, not NMS.
- Greedy cannot always return K distinct cells either. At K = 256 it averages 201–241 unique cells and pads the rest with cell 0, because a 32×32 grid with 3×3 exclusion fills up.
- **Not yet measured:** overlap on real trained F maps (`--maps` with dumped `out["F"]`), and **mAP with Method 2**. All accuracy numbers so far use the greedy sampler.

---

## 10. Exact greedy, made fast (algorithm work)

Goal: keep greedy's exact picks while matching Method 2's speed, and make the result portable to a RISC-V host core.

**Variants** (all scan F quantised to uint8; ties go to the lower raster index; each pick blocks its 3×3 neighbourhood)

| Variant | Idea |
|---|---|
| naive | full masked argmax over D² cells per pick |
| rowmax | keep the best cell per row; rescan rows r−1..r+1 after each pick |
| tilemax | keep the best cell per 8×8 tile; rescan the 1–4 tiles a pick touches |
| sortscan | counting-sort all cells once, then walk the list and skip blocked cells |
| threshold | sort only the top band of about 4K cells; move to the next band if needed |
| bucketscan | one pass builds a linked list per score; walk scores 255 → 0 with a bit-packed blocked map |
| threshold+ | threshold, plus a bitmap, a histogram fused into quantisation, and a fallback to bucketscan when one band covers more than half the grid |

**Correctness** **[re-run]**: all variants are identical to naive on every test map (3 real 64×64 maps, random, heavy-tie, all-zero and mostly-zero maps at 32² and 64²) for K ∈ {64, 128, 256}. ✅

**Memory-op cost** (a proxy for RISC-V cycles; 64×64 real-image maps) **[re-run]**

| K | Variant | Python ms | Loads | Stores | **Loads + stores** | vs naive |
|---:|---|---:|---:|---:|---:|---:|
| 64 | naive | 0.468 | 524,288 | 4,656 | 528,944 | 1× |
| | rowmax | 0.368 | 36,629 | 5,163 | 41,793 | 12.7× |
| | tilemax | 0.813 | 28,843 | 4,979 | 33,821 | 15.6× |
| | sortscan | 0.129 | 16,847 | 17,200 | 34,047 | 15.5× |
| | threshold | 0.100 | 13,942 | 10,065 | 24,007 | 22.0× |
| | bucketscan | 0.189 | 8,696 | 8,702 | 17,397 | 30.4× |
| | **threshold+** | **0.072** | 10,035 | 5,663 | **15,698** | **33.7×** |
| 128 | naive | 1.150 | 1,048,576 | 5,217 | 1,053,793 | 1× |
| | rowmax | 0.748 | 65,024 | 6,103 | 71,127 | 14.8× |
| | tilemax | 1.466 | 49,707 | 5,738 | 55,445 | 19.0× |
| | sortscan | 0.199 | 17,189 | 17,761 | 34,950 | 30.2× |
| | threshold | 0.180 | 15,341 | 11,664 | 27,005 | 39.0× |
| | **bucketscan** | 0.252 | 9,256 | 8,891 | **18,147** | **58.1×** |
| | threshold+ | **0.140** | 11,624 | 6,890 | 18,514 | 56.9× |
| 256 | naive | 2.766 | 2,097,152 | 6,344 | 2,103,496 | 1× |
| | rowmax | 1.526 | 122,069 | 7,991 | 130,060 | 16.2× |
| | tilemax | 2.763 | 90,496 | 7,246 | 97,742 | 21.5× |
| | sortscan | 0.348 | 18,287 | 18,888 | 37,174 | 56.6× |
| | threshold | 0.355 | 21,375 | 16,326 | 37,701 | 55.8× |
| | **bucketscan** | 0.389 | 10,773 | 9,272 | **20,044** | **104.9×** |
| | threshold+ | **0.313** | 18,039 | 10,806 | 28,845 | 72.9× |

Bucketscan has the lowest memory traffic at K ≥ 128, and its cost barely grows with K. Threshold+ is the fastest in Python at every K.

**Full pipeline with a host sampler** (`greedy_variants_pipeline.json`; 30 fresh images; the sampler stage includes the GPU→host→GPU transfers; bicubic positional interpolation)

*Fixation-sampler stage (ms)*

| Sampler | 512² B=1 | 512² B=4 | 1024² B=1 | 1024² B=4 |
|---|---:|---:|---:|---:|
| torch greedy (original) | 10.04 | 9.60 | 7.48 | 5.04 |
| Max-Pool (Method 2, *not exact*) | 0.63 | 0.11 | 0.16 | 0.15 |
| host naive | 1.48 | 3.56 | 1.46 | 5.47 |
| host rowmax | 1.80 | 3.90 | 1.08 | 4.22 |
| host tilemax | 2.31 | 6.68 | 2.05 | 7.03 |
| **host sortscan** | 0.91 | 1.43 | 0.83 | 1.62 |
| host threshold | 0.99 | 4.43 ⚠ | **0.66** | 1.29 |
| **host bucketscan** | **0.86** | **1.14** | 0.92 | 1.67 |
| **host threshold+** | 1.07 | 1.53 | 0.89 | **1.26** |

*End-to-end (ms)*

| Sampler | 512² B=1 | 512² B=4 | 1024² B=1 | 1024² B=4 |
|---|---:|---:|---:|---:|
| torch greedy (original) | 21.57 ± 1.53 | 35.06 | 19.59 | 40.98 |
| Max-Pool (Method 2) | 12.06 ± 0.50 | **26.15** | **12.43** | 36.91 |
| host naive | 12.86 | 29.08 | 13.51 | 41.31 |
| host rowmax | 13.39 | 29.37 | 12.75 | 40.47 |
| host tilemax | 13.68 | 32.05 | 14.62 | 43.20 |
| **host sortscan** | **12.05 ± 0.37** | 27.34 | 13.62 | 38.35 |
| host threshold | 12.32 | 35.70 ⚠ | 12.94 | 35.99 |
| **host bucketscan** | 12.17 | 26.85 | 13.38 | 36.85 |
| **host threshold+** | 12.44 | 26.51 | 13.18 | **35.96** |

⚠ The 512² B = 4 threshold run was disturbed: its CNN stage doubled to 8.6 ms and the sampler std is ±2.98 ms. Re-run it.

**Takeaways**

- At B = 1, host-side exact greedy (sortscan / bucketscan / threshold+) matches Method 2 end-to-end, within about 0.4 ms (12.05–12.44 vs 12.06 ms), with no loss of fidelity.
- At B = 4 the host samplers process images one after another, so their cost grows with B (1.1–1.5 ms vs 0.11 ms). The end-to-end cost is still only 1–5 % more than Method 2, because the ViT dominates.
- At 1024² with B = 4 the CNN (about 16 ms) dominates, and threshold / threshold+ are level with Method 2 (≈ 36.0 vs 36.9 ms; std ±0.7–2.2 ms).
- Exactness is defined on the **uint8-quantised** F. Picks can differ from the float32 torch greedy where quantisation creates ties.

---

## 11. C / RISC-V port **[re-run]**

`greedy_c/` holds naive, sortscan and bucketscan in C, tested against the golden picks in `sampler_testdata/` (3 COCO images resized to 1024² → 64×64 grid, K ∈ {64, 128, 256}). The maps currently come from a **proxy saliency**: L2 energy of ImageNet MobileNetV3 features, not the trained S/O heads.

| Build | Result |
|---|---|
| Native gcc -O2 | ✅ **27/27** match the golden picks (3 files × 3 images × 3 K values) |
| rv32im + qemu-riscv32, as committed | ❌ crashes with SIGSEGV on the first test |
| rv32im with `-fno-tree-loop-distribute-patterns` | ✅ naive 9/9 · ❌ sortscan and bucketscan still crash with SIGSEGV |

**Diagnosis of the first crash:** at -O2, GCC turns the hand-written `memset` loop in `RV32_CRT` into a call to `memset`, so `memset` calls itself until the stack overflows. Adding `-fno-tree-loop-distribute-patterns` (or `-ffreestanding` / `-fno-builtin`) to the rv32 compile fixes it. This was checked with objdump on a scratch build; no project files were changed. The remaining sortscan/bucketscan crash is not diagnosed yet. Their binaries have a BSS-only RW segment of about 13.8 KB (`order`, `start`, `blocked`), which may be worth checking against qemu-user's ELF loader or the startup code.

---

## 12. Positional-interpolation mode (stage 4b)

The sparse model samples the pretrained 14×14 ViT positional table at continuous fixation coordinates with `grid_sample(mode='bicubic')`. PyTorch runs one thread per point and loops over D = 384 channels inside it, so with K = 128 points the GPU is badly under-used. `fast_bicubic.py` rewrites this as a 16-tap weight matrix followed by a single GEMM (with `torch.compile`).

**Isolated `grid_sample`** (`interp_mode_results.json`)

| B | Bicubic (ms) | Bilinear (ms) | **Fast bicubic (ms)** | Fast vs bicubic | Bilinear vs bicubic max\|Δ\| | rel-L2 | Fast vs bicubic max\|Δ\| |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.186 | 0.125 | **0.088** | **13.5×** | 0.671 | 9.1 % | 1.2e-6 |
| 4 | 1.411 | 0.245 | **0.094** | **15.0×** | 0.781 | 9.4 % | 9.5e-7 |
| 8 | 1.494 | 0.254 | **0.142** | **10.5×** | 0.748 | 9.3 % | 1.4e-6 |
| 12 | 1.501 | 0.255 | **0.183** | **8.2×** | 0.773 | 9.1 % | 1.4e-6 |

**Inside the pipeline: stage 4b and total (ms; 50 images)**

| B | Sampler | 4b bicubic | 4b bilinear | 4b fast | Total bicubic | Total bilinear | Total fast |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Original | 1.24 | 0.16 | 0.35 | 22.26 | 20.95 | 23.82* |
| 1 | Method 2 | 1.25 | 0.15 | 0.20 | 12.12 | 11.17 | **11.15** |
| 4 | Original | 1.47 | 0.25 | 0.10 | 33.68 | 31.60 | 33.17 |
| 4 | Method 2 | 1.56 | 0.26 | 0.09 | 25.04 | 24.01 | **23.67** |
| 8 | Original | 1.59 | 0.27 | 0.14 | 58.71 | 58.79 | 60.40 |
| 8 | Method 2 | 1.66 | 0.37 | 0.21 | 56.98 | **54.45** | 56.92 |
| 12 | Original | 1.64 | 0.27 | 0.18 | 86.17 | 83.18 | 83.35 |
| 12 | Method 2 | 1.59 | 0.27 | 0.18 | 82.13 | 81.26 | **81.24** |

\* This row's noise comes from the original greedy sampler (11.79 ms in that run), not from interpolation.

**Takeaways**

- Fast bicubic gives **the same result as bicubic** and removes about 1.0–1.4 ms per batch. Method 2 with fast bicubic is the fastest exact-interpolation setup: **11.15 ms at B = 1 (89.7 FPS)**.
- Bilinear is about as fast, but it changes the positional embeddings by about 9 % (relative L2), so it would need retraining or an accuracy check. Fast bicubic has no such trade-off.

---

## 13. Measurement consistency and unverified claims

The same quantity, measured by different harnesses (512², B = 1, K = 128):

| Quantity | Values seen (ms) | Notes |
|---|---|---|
| Dense ViT total | 48.18 · 48.23 · 48.35 · 48.76 · 48.94 · (55.6 single-shot OOM run) | stable |
| Original sparse total | 19.29 · 19.57 · 19.76 · 20.07 · 20.10 · 21.57 · 22.26 · 23.66 · 24.70 · (26.3 single-shot) | spread 19–25 |
| Original sampler | 8.62 (alone) · 8.83 · 9.44 · 10.04 · 10.47 · 11.85 · 12.06 · 12.31 | launch-bound, so sensitive to CPU load |
| Method 2 total | 10.62 · 10.68 · 10.74 · 10.87 · 11.17 · 12.06 · 12.12 · 12.89 · (14.1 single-shot) | spread 10.6–12.9 |
| Method 2 sampler | 0.37 (alone) · 0.44 · 0.49 · 0.59–0.63 · 0.75 | |

The 29 Sep harnesses reuse one fixed input. The October harnesses (`native_timings`, `interp_mode`, `greedy_variants`) draw a **fresh image per run** and put CUDA-event syncs between stages, which adds about 1–2 ms. **For a paper, quote all models from one harness.** The fresh-image harness is the more realistic one, but it has no dense-ViT row yet.

**Figures in `hardware_optimization_report.tex` without a saved raw result:**

- The fidelity table (100 % / 92.4–95.3 % / 98.8 % / 68.5–74.2 %). This conflicts with §9 (b) and (c).
- Method 2 at K = 256 (15.34 ms, 1,350 kernels) and K = 64 (7.45 ms, 134.2 FPS, 1,280 kernels), and the original sampler kernel counts at K = 256 / 64 (7,650 / 2,310).
- The "30 diverse image batches, σ ≤ 0.25 ms" statement. Saved fresh-image runs show σ of 0.5–2.7 ms on totals.
- The B = 1 / 4 / 8 / 12 stage table (§6), which came from stdout of `profile_large_batch_breakdown.py`.

---

## 14. Accuracy context (from `Saliency-Objectness-Guided-Sparse-ViT/report.tex`)

MS-COCO validation, 2,476 images. **These numbers are for the greedy sampler at K = 256.**

| Epoch | Dense mAP | Dense AP50 | Dense AP75 | Sparse K=256 mAP | Sparse AP50 | Sparse AP75 |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 11.54 | 26.86 | 8.02 | 8.46 | 20.82 | 5.12 |
| **140 (same epoch)** | **12.85** | **28.65** | **9.96** | **10.16** | **23.66** | **7.12** |
| 265 (dense peak) | 16.05 | 32.89 | 13.48 | — | — | — |
| 300 | 15.64 | 31.97 | 13.33 | — | — | — |

- At epoch 140, sparse K = 256 keeps **82.6 % of the dense AP50** while using 26.5 % of the tokens.
- **Mismatch:** the latency headlines are for **K = 128 with Method 2**, but the accuracy numbers are for **K = 256 with greedy**. There is no accuracy measurement yet for K = 128, for Method 2, or for exact-greedy on the host.

---

## 15. Recommendations and open items

**Deployment picks**

| Scenario | Recommended configuration |
|---|---|
| B = 1 edge / real-time (GPU only) | Method 2 + fast bicubic → **11.15 ms**, but only after its accuracy is validated (§9) |
| B = 1 with **exact greedy semantics** | Host **sortscan / bucketscan / threshold+** + fast bicubic → about 12 ms, same picks as greedy |
| RISC-V host core next to an accelerator | **bucketscan** (fewest memory operations at K ≥ 128) or **threshold+** (fastest in Python, fewest operations at K = 64) |
| Large batch (B ≥ 8) server | Any sampler (within 2–4 %). Prefer the parallel or host-exact ones for lower CPU load and variance |
| High resolution (≥ 1024²) | Sparse with any fast sampler: about 13 ms vs 351 ms dense |

**Open items, by priority**

1. **Accuracy of Method 2** (and of K = 128), measured as mAP or AP50 against greedy. Without it, the 4.55× headline is not comparable with the accuracy table.
2. Run `analyze_sampler_overlap.py --maps` on **real trained F maps**, and regenerate `sampler_testdata` with `--checkpoint`, not the proxy saliency.
3. Re-measure OOM / max batch **under `torch.no_grad()`**.
4. Produce one consistent table with the fresh-image harness covering dense, original, Method 2 and host-exact, for K = 64 / 128 / 256. This replaces the unverified TeX figures in §13.
5. Fix the rv32im test: add `-fno-tree-loop-distribute-patterns` (or `-ffreestanding`) to the riscv compile line, then debug the sortscan/bucketscan crash under qemu. After that, collect cycle counts to set against the memory-operation model.
6. Re-run the disturbed 512² B = 4 "threshold" point.
7. Correct the TeX fidelity section (§9) before submission.
