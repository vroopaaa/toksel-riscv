import os
import sys
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Saliency-Objectness-Guided-Sparse-ViT"))
WORKSPACE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

for p in [PROJECT_DIR, WORKSPACE_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.fixation_sampler import FixationSampler
from hardware_analysis.optimized_samplers import MaxPoolFixationSampler

@torch.no_grad()
def test_sampler_scaling():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">> Running on device: {device} ({torch.cuda.get_device_name(0)})")

    orig_sampler = FixationSampler(K=128, patch_size=16, image_size=(512, 512)).to(device)
    maxpool_sampler = MaxPoolFixationSampler(K=128, patch_size=16, image_size=(512, 512)).to(device)

    batches = [1, 2, 4, 8, 12, 16]

    print("\n" + "=" * 80)
    print("      STANDALONE SAMPLER BENCHMARK (ISOLATED WITH SYNCHRONIZATION)")
    print("=" * 80)
    print(f"{'Batch Size':<12} | {'Original Sampler (ms)':<25} | {'Method 2 MaxPool (ms)':<25}")
    print("-" * 80)

    for b in batches:
        S = torch.rand(b, 1, 32, 32, device=device)
        O = torch.rand(b, 1, 32, 32, device=device)

        # Warmup
        for _ in range(5):
            _ = orig_sampler(S, O)
            _ = maxpool_sampler(S, O)
        torch.cuda.synchronize()

        # Measure Original
        t_orig = []
        for _ in range(20):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = orig_sampler(S, O)
            end.record()
            torch.cuda.synchronize()
            t_orig.append(start.elapsed_time(end))

        # Measure MaxPool
        t_pool = []
        for _ in range(20):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = maxpool_sampler(S, O)
            end.record()
            torch.cuda.synchronize()
            t_pool.append(start.elapsed_time(end))

        print(f"B = {b:<8d} | {np.mean(t_orig):8.2f} ± {np.std(t_orig):4.2f} ms     | {np.mean(t_pool):8.2f} ± {np.std(t_pool):4.2f} ms")

if __name__ == "__main__":
    test_sampler_scaling()
