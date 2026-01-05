from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import lpips


ArrayLike = Union[np.ndarray, "torch.Tensor"]


@dataclass
class LPIPSEvaluator:
    """Convenience wrapper that keeps an LPIPS network in memory."""

    net: str = "alex"
    device: Optional[str] = None
    assume_bgr: bool = True

    def __post_init__(self) -> None:
        resolved_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = resolved_device
        self._model = lpips.LPIPS(net=self.net).to(resolved_device)
        self._model.eval()

    def __call__(self, img_a: ArrayLike, img_b: ArrayLike) -> float:
        """Compute LPIPS distance between two images."""

        tensor_a = self._prepare(img_a)
        tensor_b = self._prepare(img_b)

        with torch.no_grad():
            dist = self._model(tensor_a, tensor_b)
        return float(dist.item())

    def _prepare(self, image: ArrayLike) -> "torch.Tensor":
        """Convert input image to a normalized tensor in [-1, 1]."""

        if isinstance(image, np.ndarray):
            tensor = self._from_numpy(image)
        elif torch.is_tensor(image):
            tensor = self._from_tensor(image)
        else:
            raise TypeError("Unsupported input type for LPIPS computation.")

        if tensor.max() > 1.0 or tensor.min() < 0.0:
            tensor = tensor / 255.0
        tensor = tensor.clamp(0.0, 1.0)
        tensor = tensor * 2.0 - 1.0
        return tensor.to(self._device)

    def _from_numpy(self, image: np.ndarray) -> "torch.Tensor":
        arr = np.asarray(image).astype(np.float32)
        if arr.ndim == 2:  # Grayscale image -> replicate channels
            arr = arr[:, :, None]
        if arr.ndim != 3:
            raise ValueError("Expected HxWxC numpy array.")
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.shape[2] != 3:
            raise ValueError("Input must have 3 channels after conversion.")
        if self.assume_bgr:
            arr = arr[:, :, ::-1]

        arr = arr / (255.0 if arr.max() > 1.01 else 1.0)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
        return tensor

    def _from_tensor(self, image: "torch.Tensor") -> "torch.Tensor":
        tensor = image.detach()
        if tensor.ndim == 3:  # CHW -> add batch dim
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 4 and tensor.shape[-1] == 3 and tensor.shape[1] != 3:
            tensor = tensor.permute(0, 3, 1, 2)
        if tensor.ndim != 4 or tensor.shape[1] != 3:
            raise ValueError("Tensor inputs must be in NCHW or CHW format with 3 channels.")
        if self.assume_bgr:
            tensor = tensor[:, [2, 1, 0], ...]
        return tensor.float().contiguous()


def compute_lpips(
    img_a: ArrayLike,
    img_b: ArrayLike,
    net: str = "alex",
    device: Optional[str] = None,
    assume_bgr: bool = True,
    evaluator: Optional[LPIPSEvaluator] = None,
) -> float:
    """Functional LPIPS helper for quick, one-off comparisons."""

    if evaluator is None:
        evaluator = LPIPSEvaluator(net=net, device=device, assume_bgr=assume_bgr)
    return evaluator(img_a, img_b)


__all__ = ["LPIPSEvaluator", "compute_lpips"]


if __name__ == "__main__":
    input_path = "input.jpg"
    watermarked_path = "test/watermarked.png"
    attacked_q70_path = "test/attacked_q70.jpg"
    attacked_resize_path = "test/attacked_resize.png"

    input_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    wm_img = cv2.imread(watermarked_path, cv2.IMREAD_COLOR)
    attacked_q70_img = cv2.imread(attacked_q70_path, cv2.IMREAD_COLOR)
    attacked_resize_img = cv2.imread(attacked_resize_path, cv2.IMREAD_COLOR)

    evaluator = LPIPSEvaluator(net="alex")

    print("LPIPS demo using input.jpg and test/ images")

    # input.jpg vs others
    score_input_wm = evaluator(input_img, wm_img)
    print(f"input.jpg vs watermarked.png: {score_input_wm:.4f}")

    score_input_aq = evaluator(input_img, attacked_q70_img)
    print(f"input.jpg vs attacked_q70.jpg: {score_input_aq:.4f}")

    score_input_ar = evaluator(input_img, attacked_resize_img)
    print(f"input.jpg vs attacked_resize.png: {score_input_ar:.4f}")

    # watermarked.png vs attacked images
    score_wm_aq = evaluator(wm_img, attacked_q70_img)
    print(f"watermarked.png vs attacked_q70.jpg: {score_wm_aq:.4f}")

    score_wm_ar = evaluator(wm_img, attacked_resize_img)
    print(f"watermarked.png vs attacked_resize.png: {score_wm_ar:.4f}")