from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SSIMResult:
    """Container for SSIM outputs."""

    score: float
    channel_scores: np.ndarray
    full_map: Optional[np.ndarray] = None


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    radius = size // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel_1d = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = np.outer(kernel_1d, kernel_1d)
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def _prepare_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError("SSIM expects images in HxWxC format.")
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr.astype(np.float32)


def _filter2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply 2D filter with reflect padding using cv2 when available, else torch."""

    if hasattr(cv2, "filter2D"):
        return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)

    # Torch fallback for environments where cv2 lacks filter2D
    device = "cpu"
    img_t = torch.from_numpy(image).to(device=device)
    ker_t = torch.from_numpy(kernel).to(device=device)
    if img_t.dim() == 2:
        img_t = img_t.unsqueeze(0)
    img_t = img_t.unsqueeze(0)  # (1,C,H,W)
    ker_t = ker_t.unsqueeze(0).unsqueeze(0)  # (1,1,k,k)
    if img_t.size(1) > 1:
        ker_t = ker_t.expand(img_t.size(1), 1, kernel.shape[0], kernel.shape[1])
    pad = kernel.shape[0] // 2
    img_t = F.pad(img_t, (pad, pad, pad, pad), mode="reflect")
    out = F.conv2d(img_t, ker_t, padding=0, groups=img_t.size(1))
    return out.squeeze(0).numpy()


def compute_ssim(
    img_a: np.ndarray,
    img_b: np.ndarray,
    *,
    data_range: Optional[float] = None,
    window_size: int = 11,
    sigma: float = 1.5,
    return_map: bool = False,
) -> SSIMResult | float:
    """Compute the SSIM score between two images."""

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd so the kernel is symmetric.")

    a = _prepare_image(img_a)
    b = _prepare_image(img_b)

    if a.shape != b.shape:
        raise ValueError("SSIM inputs must share the same spatial dimensions.")

    if data_range is None:
        max_val = max(float(a.max()), float(b.max()))
        data_range = 255.0 if max_val > 1.5 else 1.0

    a = a / data_range
    b = b / data_range

    kernel = _gaussian_kernel(window_size, sigma)
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    channel_scores = []
    maps = []

    for c in range(a.shape[2]):
        x = a[:, :, c]
        y = b[:, :, c]

        mu_x = cv2.filter2D(x, -1, kernel, borderType=cv2.BORDER_REFLECT)
        mu_y = cv2.filter2D(y, -1, kernel, borderType=cv2.BORDER_REFLECT)

        sigma_x = cv2.filter2D(x * x, -1, kernel, borderType=cv2.BORDER_REFLECT) - mu_x ** 2
        sigma_y = cv2.filter2D(y * y, -1, kernel, borderType=cv2.BORDER_REFLECT) - mu_y ** 2
        sigma_xy = cv2.filter2D(x * y, -1, kernel, borderType=cv2.BORDER_REFLECT) - mu_x * mu_y

        numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        ssim_map = numerator / (denominator + 1e-12)

        channel_scores.append(float(ssim_map.mean()))
        if return_map:
            maps.append(ssim_map[:, :, None])

    mean_score = float(np.mean(channel_scores))
    if not return_map:
        return mean_score

    full_map = np.concatenate(maps, axis=2) if maps else None
    return SSIMResult(score=mean_score, channel_scores=np.array(channel_scores), full_map=full_map)


__all__ = ["compute_ssim", "SSIMResult"]


if __name__ == "__main__":
    input_path = "input.jpg"
    watermarked_path = "test/watermarked.png"
    attacked_q70_path = "test/attacked_q70.jpg"
    attacked_resize_path = "test/attacked_resize.png"

    input_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    wm_img = cv2.imread(watermarked_path, cv2.IMREAD_COLOR)
    attacked_q70_img = cv2.imread(attacked_q70_path, cv2.IMREAD_COLOR)
    attacked_resize_img = cv2.imread(attacked_resize_path, cv2.IMREAD_COLOR)

    loaded = {
        "input": input_img,
        "watermarked": wm_img,
        "attacked_q70": attacked_q70_img,
        "attacked_resize": attacked_resize_img,
    }
    missing = [name for name, img in loaded.items() if img is None]
    if missing:
        print("SSIM demo skipped because files are missing:", ", ".join(missing))
        exit(0)

    print("SSIM demo using input.jpg and test/ images")

    # input.jpg vs others
    score_wm_ar = compute_ssim(input_img, input_img)
    print(f"input.png vs input.png: {score_wm_ar:.4f}")

    score_input_wm = compute_ssim(input_img, wm_img)
    print(f"input.jpg vs watermarked.png: {score_input_wm:.4f}")

    score_input_aq = compute_ssim(input_img, attacked_q70_img)
    print(f"input.jpg vs attacked_q70.jpg: {score_input_aq:.4f}")

    score_input_ar = compute_ssim(input_img, attacked_resize_img)
    print(f"input.jpg vs attacked_resize.png: {score_input_ar:.4f}")

    # watermarked.png vs attacked images
    score_wm_aq = compute_ssim(wm_img, attacked_q70_img)
    print(f"watermarked.png vs attacked_q70.jpg: {score_wm_aq:.4f}")

    score_wm_ar = compute_ssim(wm_img, attacked_resize_img)
    print(f"watermarked.png vs attacked_resize.png: {score_wm_ar:.4f}")


