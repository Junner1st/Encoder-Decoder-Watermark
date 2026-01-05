from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.utils.SSIM import compute_ssim as ssim_metric


def _extract_images(batch: Any) -> torch.Tensor:
    """Return the image tensor from common dataset batch structures."""

    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (list, tuple)):
        for item in batch:
            if isinstance(item, torch.Tensor):
                return item
    if isinstance(batch, dict):
        for key in ("image", "images", "input", "inputs", "x"):
            if key in batch and isinstance(batch[key], torch.Tensor):
                return batch[key]
    raise TypeError("Unable to extract image tensor from batch. Provide Tensor, tuple, or dict with images.")


def _tensor_to_numpy(img: torch.Tensor) -> np.ndarray:
    array = img.detach().cpu().clamp(0.0, 1.0).numpy()
    if array.ndim == 4:
        array = array[0]
    return np.transpose(array, (1, 2, 0))


def evaluate_autoencoder(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
    *,
    max_batches: Optional[int] = None,
    compute_ssim_score: bool = False,
    compute_lpips_score: bool = False,
    lpips_net: str = "alex",
) -> Dict[str, float]:
    """Run the model on a loader and aggregate reconstruction metrics."""

    model.eval()
    device = torch.device(device)

    total_mse = 0.0
    total_mae = 0.0
    total_elements = 0
    ssim_scores: list[float] = []
    lpips_scores: list[float] = []

    lpips_evaluator = None
    if compute_lpips_score:
        from src.utils.LPIPS import LPIPSEvaluator  # Imported lazily to avoid dependency when unused

        lpips_evaluator = LPIPSEvaluator(net=lpips_net, assume_bgr=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = _extract_images(batch).to(device, non_blocking=True)
            recon = model(images)

            total_mse += F.mse_loss(recon, images, reduction="sum").item()
            total_mae += F.l1_loss(recon, images, reduction="sum").item()
            total_elements += recon.numel()

            if compute_ssim_score or compute_lpips_score:
                images_cpu = images.cpu()
                recon_cpu = recon.cpu()
                batch_size = images_cpu.size(0)

                if compute_ssim_score:
                    for idx in range(batch_size):
                        ref = _tensor_to_numpy(images_cpu[idx])
                        rec = _tensor_to_numpy(recon_cpu[idx])
                        ssim_scores.append(float(ssim_metric(ref, rec)))

                if compute_lpips_score and lpips_evaluator is not None:
                    for idx in range(batch_size):
                        lpips_scores.append(float(lpips_evaluator(images_cpu[idx], recon_cpu[idx])))

    if total_elements == 0:
        raise RuntimeError("Dataloader produced zero elements; cannot compute metrics.")

    mean_mse = total_mse / total_elements
    mean_mae = total_mae / total_elements
    psnr = 10.0 * math.log10(1.0 / max(mean_mse, 1e-12))

    results: Dict[str, float] = {
        "mse": mean_mse,
        "mae": mean_mae,
        "psnr": psnr,
    }

    if ssim_scores:
        results["ssim"] = float(np.mean(ssim_scores))
    if lpips_scores:
        results["lpips"] = float(np.mean(lpips_scores))

    return results
