#!/usr/bin/env python3
"""
Optimized Fixation Samplers:
Method 1: Direct 2D Grid Slicing (User's intuition - local 3x3 window zeroing without distance matrix).
Method 2: Max-Pool Morphological NMS (Zero-loop parallel peak detection).

Does NOT modify any original files in models/.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from typing import Tuple, Optional


class GridSlicingFixationSampler(nn.Module):
    """
    Method 1: Direct 2D Grid Slicing.
    Eliminates the 1024x1024 distance matrix entirely.
    Directly zeroes out the local 3x3 window around the selected cell on the 2D grid.
    """
    def __init__(
        self,
        K: int = 128,
        alpha: float = 0.60,
        beta: float = 0.40,
        patch_size: int = 16,
        image_size: Tuple[int, int] = (512, 512)
    ):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.patch_size = patch_size
        self.image_size = image_size

    def normalize_map(self, M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        B = M.shape[0]
        M_flat = M.view(B, -1)
        M_min = M_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        M_max = M_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        return (M - M_min) / (M_max - M_min + eps)

    def forward(
        self,
        S: torch.Tensor,
        O: torch.Tensor,
        image_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, H_feat, W_feat = S.shape
        H_img, W_img = image_size if image_size is not None else self.image_size

        S_hat = self.normalize_map(S)
        O_hat = self.normalize_map(O)
        F = self.alpha * S_hat + self.beta * O_hat  # (B, 1, H', W')

        stride_y = H_img / H_feat
        stride_x = W_img / W_feat

        # Keep scores as a 2D grid: (B, H_feat, W_feat)
        scores_2d = F.squeeze(1).clone()

        pts_list = []
        half_p = self.patch_size / 2.0
        min_x, max_x = half_p, W_img - half_p
        min_y, max_y = half_p, H_img - half_p

        for _ in range(self.K):
            # 1. Flatten only to find the single best peak
            flat_scores = scores_2d.view(B, -1)
            best_idx = torch.argmax(flat_scores, dim=-1)  # (B,)

            # 2. Convert index directly to (row, col) grid coordinates
            best_r = best_idx // W_feat  # (B,)
            best_c = best_idx % W_feat   # (B,)

            # Center coordinates in pixel space
            px = (best_c.float() + 0.5) * stride_x
            py = (best_r.float() + 0.5) * stride_y
            clamped_x = torch.clamp(px, min_x, max_x)
            clamped_y = torch.clamp(py, min_y, max_y)
            pts_list.append(torch.stack([clamped_x, clamped_y], dim=-1))

            # 3. Direct 2D window zeroing (suppress adjacent 3x3 cells without distance matrix!)
            for b in range(B):
                r = best_r[b].item()
                c = best_c[b].item()
                r_start, r_end = max(0, r - 1), min(H_feat, r + 2)
                c_start, c_end = max(0, c - 1), min(W_feat, c + 2)
                scores_2d[b, r_start:r_end, c_start:c_end] = -1e4

        coords = torch.stack(pts_list, dim=1)  # (B, K, 2)

        norm_coords = torch.stack([
            coords[:, :, 0] / W_img,
            coords[:, :, 1] / H_img
        ], dim=-1)

        sample_grid_x = 2.0 * norm_coords[..., 0] - 1.0
        sample_grid_y = 2.0 * norm_coords[..., 1] - 1.0
        sample_grid = torch.stack([sample_grid_x, sample_grid_y], dim=-1).view(B, self.K, 1, 2)

        fixation_scores = F_nn.grid_sample(
            F, sample_grid, mode='bilinear', padding_mode='border', align_corners=False
        )
        fixation_scores = fixation_scores.squeeze(-1).permute(0, 2, 1)

        return coords, norm_coords, fixation_scores, F


class MaxPoolFixationSampler(nn.Module):
    """
    Method 2: Max-Pool Morphological NMS.
    Zero Python loops! Completely vectorized in a single parallel GPU kernel.
    Uses 3x3 MaxPool to identify all local spatial peaks, then selects top-K.
    """
    def __init__(
        self,
        K: int = 128,
        alpha: float = 0.60,
        beta: float = 0.40,
        patch_size: int = 16,
        image_size: Tuple[int, int] = (512, 512)
    ):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.patch_size = patch_size
        self.image_size = image_size

    def normalize_map(self, M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        B = M.shape[0]
        M_flat = M.view(B, -1)
        M_min = M_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        M_max = M_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        return (M - M_min) / (M_max - M_min + eps)

    def forward(
        self,
        S: torch.Tensor,
        O: torch.Tensor,
        image_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, H_feat, W_feat = S.shape
        H_img, W_img = image_size if image_size is not None else self.image_size

        S_hat = self.normalize_map(S)
        O_hat = self.normalize_map(O)
        F = self.alpha * S_hat + self.beta * O_hat  # (B, 1, H', W')

        # 1. Parallel 2D Max-Pooling (kernel_size=3, stride=1, padding=1)
        # Finds whether each cell is greater than or equal to its 3x3 local neighborhood
        local_max = F_nn.max_pool2d(F, kernel_size=3, stride=1, padding=1)
        is_peak = (F == local_max)

        # 2. Prioritize local peaks over non-peaks without looping
        # True peaks keep their full score; non-peaks are dampened by 1e-4 so peaks always rank first
        filtered_F = torch.where(is_peak, F, F * 1e-4)

        # 3. Parallel Top-K selection in a SINGLE CUDA instruction!
        flat_filtered = filtered_F.view(B, -1)
        _, topk_idx = torch.topk(flat_filtered, k=self.K, dim=-1)  # (B, K)

        # 4. Map indices to pixel coordinates in parallel
        stride_y = H_img / H_feat
        stride_x = W_img / W_feat

        best_r = topk_idx // W_feat  # (B, K)
        best_c = topk_idx % W_feat   # (B, K)

        px = (best_c.float() + 0.5) * stride_x
        py = (best_r.float() + 0.5) * stride_y

        half_p = self.patch_size / 2.0
        clamped_x = torch.clamp(px, half_p, W_img - half_p)
        clamped_y = torch.clamp(py, half_p, H_img - half_p)
        coords = torch.stack([clamped_x, clamped_y], dim=-1)  # (B, K, 2)

        norm_coords = torch.stack([
            coords[:, :, 0] / W_img,
            coords[:, :, 1] / H_img
        ], dim=-1)

        sample_grid_x = 2.0 * norm_coords[..., 0] - 1.0
        sample_grid_y = 2.0 * norm_coords[..., 1] - 1.0
        sample_grid = torch.stack([sample_grid_x, sample_grid_y], dim=-1).view(B, self.K, 1, 2)

        fixation_scores = F_nn.grid_sample(
            F, sample_grid, mode='bilinear', padding_mode='border', align_corners=False
        )
        fixation_scores = fixation_scores.squeeze(-1).permute(0, 2, 1)

        return coords, norm_coords, fixation_scores, F
