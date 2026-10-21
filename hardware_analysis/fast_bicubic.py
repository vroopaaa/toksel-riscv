#!/usr/bin/env python3
"""
Channel-parallel bicubic positional interpolation.

Drop-in replacement for
    F.grid_sample(table, grid, mode='bicubic', padding_mode='border', align_corners=False)
when sampling K points from a small (G x G x D) positional-embedding table.

Why: PyTorch's grid_sampler_2d_kernel launches one thread per sample point and
loops over all D channels inside the thread. With K=128 points that is only
128 threads x (384 channels x 16 taps) of serial work -> ~1.3-1.5 ms on a GTX 1650.

Here each point's bicubic output is written as a weighted sum of 16 table rows:
    1. compute the 4 x-taps / 4 y-taps and their Keys cubic weights   -> (B, K, 16)
    2. scatter them into a dense (B, K, G*G) weight matrix
    3. one batched GEMM with the (G*G, D) table                       -> (B, K, D)
The GEMM parallelises over K and D, so the GPU is fully used.

Numerically identical to grid_sample (same unnormalize, same A=-0.75 coefficients,
same per-tap border clamp), and differentiable w.r.t. both the table and the coords.

Does NOT modify any original files in models/.
"""

import torch
import torch.nn as nn

CUBIC_A = -0.75  # PyTorch's (and OpenCV's) Keys bicubic constant


def _cubic_weights(t: torch.Tensor) -> torch.Tensor:
    """Keys cubic convolution weights for taps at offsets -1, 0, +1, +2. t: (...,) in [0, 1) -> (..., 4)."""
    A = CUBIC_A

    def cc1(x):  # |x| <= 1
        return ((A + 2) * x - (A + 3)) * x * x + 1

    def cc2(x):  # 1 < |x| < 2
        return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A

    return torch.stack([cc2(t + 1), cc1(t), cc1(1 - t), cc2(2 - t)], dim=-1)


def bicubic_weight_matrix(norm_coords: torch.Tensor, G: int) -> torch.Tensor:
    """
    Args:
        norm_coords: (B, K, 2) in [0, 1], (x, y)
        G: grid size of the square positional table
    Returns:
        W: (B, K, G*G) such that W @ table(G*G, D) == bicubic grid_sample output
    """
    B, K, _ = norm_coords.shape
    # [0,1] -> [-1,1] -> pixel space, exactly as grid_sample with align_corners=False
    g = 2.0 * norm_coords - 1.0
    pix = ((g + 1.0) * G - 1.0) / 2.0                     # (B, K, 2)

    base = torch.floor(pix)
    t = pix - base                                        # (B, K, 2)
    w = _cubic_weights(t)                                 # (B, K, 2, 4)

    offsets = torch.arange(-1, 3, device=pix.device, dtype=pix.dtype)
    taps = (base.unsqueeze(-1) + offsets).clamp_(0, G - 1).long()  # (B, K, 2, 4) border padding

    wx, wy = w[:, :, 0], w[:, :, 1]                       # (B, K, 4)
    ix, iy = taps[:, :, 0], taps[:, :, 1]                 # (B, K, 4)

    weights = (wy.unsqueeze(-1) * wx.unsqueeze(-2)).reshape(B, K, 16)
    flat_idx = (iy.unsqueeze(-1) * G + ix.unsqueeze(-2)).reshape(B, K, 16)

    # scatter_add: clamped border taps can hit the same cell more than once
    W = torch.zeros(B, K, G * G, device=pix.device, dtype=pix.dtype)
    return W.scatter_add_(-1, flat_idx, weights)


def fast_bicubic_pos_interp(norm_coords: torch.Tensor, grid_pos_embed: torch.Tensor) -> torch.Tensor:
    """
    Args:
        norm_coords: (B, K, 2) in [0, 1]
        grid_pos_embed: (1, G*G, D) pretrained ViT positional table (no cls token)
    Returns:
        (B, K, D) identical to the bicubic grid_sample path in PositionalInterpolator
    """
    G = int(round(grid_pos_embed.shape[1] ** 0.5))
    W = bicubic_weight_matrix(norm_coords, G)             # (B, K, G*G)
    return torch.matmul(W, grid_pos_embed[0])             # (B, K, D)


class FastBicubicPositionalInterpolator(nn.Module):
    """
    Same interface and parameters as models.positional_encoding.PositionalInterpolator,
    with the bicubic grid_sample replaced by the GEMM formulation.
    Shares (not copies) the Fourier encoder and alpha of the wrapped interpolator,
    so trained checkpoints work unchanged:
        model.pos_interpolator = FastBicubicPositionalInterpolator(model.pos_interpolator)
    """
    def __init__(self, base_interpolator: nn.Module):
        super().__init__()
        self.embed_dim = base_interpolator.embed_dim
        self.grid_size = base_interpolator.grid_size
        self.fourier_encoder = base_interpolator.fourier_encoder
        self.alpha = base_interpolator.alpha

    def forward(self, norm_coords: torch.Tensor, pretrained_grid_pos_embed: torch.Tensor) -> torch.Tensor:
        interpolated = fast_bicubic_pos_interp(norm_coords, pretrained_grid_pos_embed)
        return interpolated + self.alpha * self.fourier_encoder(norm_coords)
