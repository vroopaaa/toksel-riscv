import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from typing import Tuple, Optional


class FixationSamplerNMS(nn.Module):
    """
    Zero-Loop Morphological Max-Pool Non-Maximum Suppression (NMS) Fixation Sampler.
    
    Replaces the sequential O(K * N^2) distance-matrix greedy loop with a 
    fully-parallel 2D morphological max-pooling peak detector and single Top-K selection.
    
    Attributes:
        K: Number of fixation coordinates to sample (default 128)
        alpha: Weight for normalized Saliency map (default 0.60)
        beta: Weight for normalized Objectness map (default 0.40)
        patch_size: Square patch size in pixels (default 16)
        image_size: (H_img, W_img) tuple (default 512, 512)
        pool_kernel: Kernel size for morphological local peak extraction (default 3)
    """
    def __init__(
        self,
        K: int = 128,
        alpha: float = 0.60,
        beta: float = 0.40,
        patch_size: int = 16,
        image_size: Tuple[int, int] = (512, 512),
        pool_kernel: int = 3
    ):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.patch_size = patch_size
        self.image_size = image_size
        self.pool_kernel = pool_kernel

    def normalize_map(self, M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Min-Max normalization per image sample: shape (B, 1, H', W')"""
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
        """
        Args:
            S: Saliency map (B, 1, H', W')
            O: Objectness map (B, 1, H', W')
            image_size: Optional (H_img, W_img) override for input image resolution
        Returns:
            coords: Pixel coordinates of K fixation points (B, K, 2) in (x, y) format
            norm_coords: Normalized coordinates in [0, 1] of shape (B, K, 2)
            fixation_scores: Differentiable continuous fixation confidence scores (B, K, 1)
            F: Combined fixation score map (B, 1, H', W')
        """
        B, _, H_feat, W_feat = S.shape
        H_img, W_img = image_size if image_size is not None else self.image_size

        # 1. Normalize and linearly combine saliency + objectness
        S_hat = self.normalize_map(S)
        O_hat = self.normalize_map(O)
        F = self.alpha * S_hat + self.beta * O_hat  # (B, 1, H', W')

        # 2. Parallel 2D Max-Pooling Morphological Dilation
        # Identifies all local spatial peak maxima across the entire grid simultaneously
        padding = self.pool_kernel // 2
        local_max = F_nn.max_pool2d(F, kernel_size=self.pool_kernel, stride=1, padding=padding)
        is_peak = (F == local_max)

        # 3. Prioritize local peak maxima over non-peaks without looping
        # True peaks retain their full fused scores; non-peaks are dampened by 1e-4
        filtered_F = torch.where(is_peak, F, F * 1e-4)

        # 4. Vectorized parallel Top-K selection in a single GPU operation
        flat_filtered = filtered_F.view(B, -1)
        _, topk_idx = torch.topk(flat_filtered, k=self.K, dim=-1)  # (B, K)

        # 5. Map top-K 1D indices to 2D continuous pixel coordinates in parallel
        stride_y = H_img / H_feat
        stride_x = W_img / W_feat

        best_r = topk_idx // W_feat  # (B, K)
        best_c = topk_idx % W_feat   # (B, K)

        px = (best_c.float() + 0.5) * stride_x
        py = (best_r.float() + 0.5) * stride_y

        # Clamp patch centers to ensure patches remain fully inside image borders
        half_p = self.patch_size / 2.0
        clamped_x = torch.clamp(px, half_p, W_img - half_p)
        clamped_y = torch.clamp(py, half_p, H_img - half_p)
        coords = torch.stack([clamped_x, clamped_y], dim=-1)  # (B, K, 2) in pixel space

        # Normalized coordinates in [0, 1]
        norm_coords = torch.stack([
            coords[:, :, 0] / W_img,
            coords[:, :, 1] / H_img
        ], dim=-1)  # (B, K, 2)

        # 6. Differentiable continuous score sampling via grid_sample
        sample_grid_x = 2.0 * norm_coords[..., 0] - 1.0
        sample_grid_y = 2.0 * norm_coords[..., 1] - 1.0
        sample_grid = torch.stack([sample_grid_x, sample_grid_y], dim=-1).view(B, self.K, 1, 2)

        fixation_scores = F_nn.grid_sample(
            F, sample_grid, mode='bilinear', padding_mode='border', align_corners=False
        )  # (B, 1, K, 1)
        fixation_scores = fixation_scores.squeeze(-1).permute(0, 2, 1)  # (B, K, 1)

        return coords, norm_coords, fixation_scores, F


# Alias for backward compatibility
MaxPoolFixationSampler = FixationSamplerNMS
