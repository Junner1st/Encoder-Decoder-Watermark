from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF


@dataclass
class AttackParams:
    jpeg_qualities: Iterable[int]
    resize_min: float
    resize_max: float
    crop_min: float
    crop_max: float
    rotate_degrees: Iterable[float]
    blur_sigma_min: float
    blur_sigma_max: float
    noise_sigma_min: float
    noise_sigma_max: float
    color_jitter: Dict[str, float]
    recompress: bool = True


def _rand_uniform(low: float, high: float, device: torch.device) -> torch.Tensor:
    return torch.empty((), device=device).uniform_(low, high)


def _diff_round(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x) + (x - x.detach())


def _dct_matrix(device: torch.device, n: int = 8) -> torch.Tensor:
    x = torch.arange(n, device=device).float()
    mat = torch.zeros((n, n), device=device)
    mat[0] = 1.0 / math.sqrt(n)
    for k in range(1, n):
        mat[k] = math.sqrt(2.0 / n) * torch.cos((math.pi * (2 * x + 1) * k) / (2 * n))
    return mat


def _quality_to_scale(q: float) -> float:
    q = max(1.0, min(100.0, q))
    if q < 50:
        return 5000.0 / q
    return 200.0 - 2.0 * q


def _quant_tables(device: torch.device, quality: float) -> torch.Tensor:
    base = torch.tensor(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ],
        device=device,
        dtype=torch.float32,
    )
    scale = _quality_to_scale(quality)
    table = torch.floor((base * scale + 50.0) / 100.0).clamp(min=1.0, max=255.0)
    return table


def jpeg_simulate(x: torch.Tensor, quality: float, differentiable: bool = True) -> torch.Tensor:
    """Approximate JPEG with straight-through quantization for backprop."""

    if x.min() < -0.1 or x.max() > 1.1:
        raise ValueError("jpeg_simulate expects inputs in [0, 1]")

    device = x.device
    B, C, H, W = x.shape
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    x = x * 255.0 - 128.0

    dct_mat = _dct_matrix(device)
    q_table = _quant_tables(device, quality).view(1, 1, 8, 8)

    blocks = x.unfold(2, 8, 8).unfold(3, 8, 8)
    # blocks shape: (B, C, H/8, W/8, 8, 8)
    flat = blocks.contiguous().view(-1, 8, 8)
    dcted = torch.matmul(dct_mat, torch.matmul(flat, dct_mat.t()))
    # q_table: (1,1,8,8) -> broadcast to (N,1,8,8) then squeeze to (N,8,8)
    q = q_table.expand(dcted.size(0), 1, 8, 8).squeeze(1)
    dcted = dcted / q
    dcted = _diff_round(dcted) if differentiable else torch.round(dcted)
    dcted = dcted * q
    idct = torch.matmul(dct_mat.t(), torch.matmul(dcted, dct_mat))
    idct = idct.view(B, C, blocks.size(2), blocks.size(3), 8, 8)
    idct = (
        idct.permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .view(B, C, x.size(2), x.size(3))
    )

    if pad_h or pad_w:
        idct = idct[:, :, :H, :W]
    out = torch.clamp((idct + 128.0) / 255.0, 0.0, 1.0)
    return out


def random_resize(imgs: torch.Tensor, params: AttackParams, size: int) -> torch.Tensor:
    scale = float(_rand_uniform(params.resize_min, params.resize_max, imgs.device))
    new_h = max(8, int(round(imgs.size(2) * scale)))
    new_w = max(8, int(round(imgs.size(3) * scale)))
    resized = F.interpolate(imgs, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return F.interpolate(resized, size=(size, size), mode="bilinear", align_corners=False)


def random_crop(imgs: torch.Tensor, params: AttackParams, size: int) -> torch.Tensor:
    crop_ratio = float(_rand_uniform(params.crop_min, params.crop_max, imgs.device))
    crop_h = max(1, int(round(imgs.size(2) * (1.0 - crop_ratio))))
    crop_w = max(1, int(round(imgs.size(3) * (1.0 - crop_ratio))))
    top = int(_rand_uniform(0, max(1, imgs.size(2) - crop_h + 1), imgs.device))
    left = int(_rand_uniform(0, max(1, imgs.size(3) - crop_w + 1), imgs.device))
    cropped = imgs[:, :, top : top + crop_h, left : left + crop_w]
    return F.interpolate(cropped, size=(size, size), mode="bilinear", align_corners=False)


def random_rotate(imgs: torch.Tensor, params: AttackParams) -> torch.Tensor:
    degs = list(params.rotate_degrees)
    if not degs:
        return imgs
    angle = float(degs[torch.randint(0, len(degs), (1,), device=imgs.device)])
    if torch.rand((), device=imgs.device) < 0.5:
        angle = -angle
    return TF.rotate(imgs, angle=angle, interpolation=TF.InterpolationMode.BILINEAR, expand=False)


def random_blur(imgs: torch.Tensor, params: AttackParams) -> torch.Tensor:
    sigma = float(_rand_uniform(params.blur_sigma_min, params.blur_sigma_max, imgs.device))
    # kernel size as odd integer based on sigma
    k = int(max(3, 2 * round(3 * sigma) + 1))
    return TF.gaussian_blur(imgs, kernel_size=k, sigma=sigma)


def random_noise(imgs: torch.Tensor, params: AttackParams) -> torch.Tensor:
    sigma = float(_rand_uniform(params.noise_sigma_min, params.noise_sigma_max, imgs.device))
    noise = torch.randn_like(imgs) * sigma
    return torch.clamp(imgs + noise, 0.0, 1.0)


def random_color_jitter(imgs: torch.Tensor, params: AttackParams) -> torch.Tensor:
    cj = params.color_jitter or {}
    brightness = cj.get("brightness", 0.0)
    contrast = cj.get("contrast", 0.0)
    saturation = cj.get("saturation", 0.0)
    hue = cj.get("hue", 0.0)
    b = float(_rand_uniform(max(0.0, 1.0 - brightness), 1.0 + brightness, imgs.device)) if brightness > 0 else 1.0
    c = float(_rand_uniform(max(0.0, 1.0 - contrast), 1.0 + contrast, imgs.device)) if contrast > 0 else 1.0
    s = float(_rand_uniform(max(0.0, 1.0 - saturation), 1.0 + saturation, imgs.device)) if saturation > 0 else 1.0
    h = float(_rand_uniform(-hue, hue, imgs.device)) if hue > 0 else 0.0
    out = TF.adjust_brightness(imgs, b)
    out = TF.adjust_contrast(out, c)
    out = TF.adjust_saturation(out, s)
    out = TF.adjust_hue(out, h)
    return torch.clamp(out, 0.0, 1.0)


def random_jpeg(imgs: torch.Tensor, params: AttackParams, differentiable: bool = True) -> torch.Tensor:
    qualities = list(params.jpeg_qualities)
    if not qualities:
        return imgs
    q = float(qualities[torch.randint(0, len(qualities), (1,), device=imgs.device)])
    return jpeg_simulate(imgs, quality=q, differentiable=differentiable)


def recompress_chain(imgs: torch.Tensor, params: AttackParams, size: int, differentiable: bool = False) -> torch.Tensor:
    y = random_jpeg(imgs, params, differentiable=differentiable)
    y = random_resize(y, params, size=size)
    y = random_jpeg(y, params, differentiable=differentiable)
    return y


def apply_stage_attacks(
    imgs: torch.Tensor,
    params: AttackParams,
    *,
    size: int,
    stage: int,
    differentiable: bool = True,
) -> torch.Tensor:
    """Curriculum attacks: stage0 none, stage1 light diff, stage2 +jpeg, stage3 +geometry."""

    y = imgs
    if stage == 0:
        return y

    # Stage 1: light differentiable attacks
    if stage >= 1:
        if torch.rand((), device=imgs.device) < 0.7:
            y = random_resize(y, params, size=size)
        if torch.rand((), device=imgs.device) < 0.5:
            y = random_blur(y, params)
        if torch.rand((), device=imgs.device) < 0.7:
            y = random_noise(y, params)
        if torch.rand((), device=imgs.device) < 0.7:
            y = random_color_jitter(y, params)

    # Stage 2: add differentiable JPEG
    if stage >= 2:
        if torch.rand((), device=imgs.device) < 0.9:
            y = random_jpeg(y, params, differentiable=differentiable)

    # Stage 3: add geometry & recompression
    if stage >= 3:
        if torch.rand((), device=imgs.device) < 0.7:
            y = random_crop(y, params, size=size)
        if torch.rand((), device=imgs.device) < 0.7:
            y = random_rotate(y, params)
        if params.recompress and torch.rand((), device=imgs.device) < 0.6:
            y = recompress_chain(y, params, size=size, differentiable=False)

    return torch.clamp(y, 0.0, 1.0)


@dataclass
class AttackScenario:
    name: str
    fn: Callable[[torch.Tensor], torch.Tensor]


def build_eval_scenarios(params: AttackParams, size: int) -> List[AttackScenario]:
    scenarios: List[AttackScenario] = []

    def make_jpeg(q: float) -> AttackScenario:
        return AttackScenario(name=f"jpeg_q{int(q)}", fn=lambda x, q=q: jpeg_simulate(x, q, differentiable=False))

    for q in params.jpeg_qualities:
        scenarios.append(make_jpeg(float(q)))

    # Crop sweep
    for ratio in [0.05, 0.1, 0.2, 0.3]:
        def _crop(x, r=ratio):
            h, w = x.size(2), x.size(3)
            crop_h = max(1, int(round(h * (1.0 - r))))
            crop_w = max(1, int(round(w * (1.0 - r))))
            top = (h - crop_h) // 2
            left = (w - crop_w) // 2
            cropped = x[:, :, top : top + crop_h, left : left + crop_w]
            return F.interpolate(cropped, size=(size, size), mode="bilinear", align_corners=False)

        scenarios.append(AttackScenario(name=f"crop_{int(ratio*100)}pct", fn=_crop))

    # Rotate sweep
    for deg in params.rotate_degrees:
        def _rot(x, d=deg):
            return TF.rotate(x, angle=d, interpolation=TF.InterpolationMode.BILINEAR, expand=False)
        scenarios.append(AttackScenario(name=f"rot_{int(deg)}deg", fn=_rot))

    # Combined harsh pipeline
    def combo(x: torch.Tensor) -> torch.Tensor:
        y = random_resize(x, params, size=size)
        y = random_jpeg(y, params, differentiable=False)
        y = random_crop(y, params, size=size)
        y = random_rotate(y, params)
        y = random_jpeg(y, params, differentiable=False)
        return y

    scenarios.append(AttackScenario(name="combo_resize_jpeg_crop_rot", fn=combo))
    return scenarios


__all__ = [
    "AttackParams",
    "apply_stage_attacks",
    "jpeg_simulate",
    "build_eval_scenarios",
    "random_resize",
    "random_crop",
    "random_rotate",
    "random_blur",
    "random_noise",
    "random_color_jitter",
    "random_jpeg",
    "recompress_chain",
]
