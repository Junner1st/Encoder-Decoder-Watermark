from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class AutoencoderBackboneConfig:
    """Configuration for the FiLM-conditioned U-Net style autoencoder."""

    in_channels: int = 3
    base_channels: int = 64
    num_scales: int = 4


class FiLMConditioner(nn.Module):
    """Maps a binary codeword into per-scale FiLM parameters."""

    def __init__(self, code_bits: int, channels_per_scale: Sequence[int], hidden: int = 512) -> None:
        super().__init__()
        self.code_bits = code_bits
        self.channels_per_scale = list(channels_per_scale)
        total_channels = sum(self.channels_per_scale)
        self.mlp = nn.Sequential(
            nn.Linear(code_bits, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * total_channels),
        )

    def forward(self, code: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        # code: (B, L) in {0,1}. Map to +/-1 for richer signal
        conditioned = (code.float() * 2.0) - 1.0
        film = self.mlp(conditioned)
        params: List[Tuple[torch.Tensor, torch.Tensor]] = []
        offset = 0
        for ch in self.channels_per_scale:
            gamma = film[:, offset : offset + ch]
            beta = film[:, offset + ch : offset + 2 * ch]
            params.append((gamma, beta))
            offset += 2 * ch
        return params


class _DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor | None,
        film: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        if film is not None:
            gamma, beta = film
            x = gamma.unsqueeze(-1).unsqueeze(-1) * x + beta.unsqueeze(-1).unsqueeze(-1)
        return x


class WatermarkAutoencoder(nn.Module):
    """U-Net backbone with FiLM conditioning for watermark injection."""

    def __init__(self, cfg: AutoencoderBackboneConfig, code_bits: int) -> None:
        super().__init__()
        self.cfg = cfg

        channels: List[int] = [cfg.base_channels * (2**i) for i in range(cfg.num_scales)]

        downs = []
        in_ch = cfg.in_channels
        for ch in channels:
            downs.append(_DownBlock(in_ch, ch))
            in_ch = ch
        self.encoder = nn.ModuleList(downs)

        ups = []
        reversed_channels = list(reversed(channels))
        for idx, ch in enumerate(reversed_channels):
            skip_ch = reversed_channels[idx + 1] if idx + 1 < len(reversed_channels) else 0
            next_ch = reversed_channels[idx + 1] if idx + 1 < len(reversed_channels) else ch
            ups.append(_UpBlock(in_ch, skip_ch=skip_ch, out_ch=next_ch))
            in_ch = next_ch
        self.decoder = nn.ModuleList(ups)

        self.out_conv = nn.Conv2d(channels[0], cfg.in_channels, kernel_size=3, padding=1)

        # FiLM conditioner matches decoder feature channels (excluding final output layer)
        decoder_channels = [blk.conv[0].out_channels for blk in self.decoder]
        self.conditioner = FiLMConditioner(code_bits=code_bits, channels_per_scale=decoder_channels)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, code_bits: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        h = x
        for block in self.encoder:
            h = block(h)
            skips.append(h)

        film_params = self.conditioner(code_bits)
        h_out = h
        for idx, block in enumerate(self.decoder):
            skip = skips[-(idx + 2)] if (idx + 1) < len(skips) else None
            film = film_params[idx] if idx < len(film_params) else None
            h_out = block(h_out, skip, film)

        recon = torch.sigmoid(self.out_conv(h_out))
        return recon


def frequency_regularizer(x: torch.Tensor) -> torch.Tensor:
    """Encourage smoothness by penalizing Laplacian energy (heuristic)."""

    laplace_kernel = x.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).view(1, 1, 3, 3)
    if x.size(1) > 1:
        laplace_kernel = laplace_kernel.expand(x.size(1), 1, 3, 3)
    filtered = F.conv2d(x, laplace_kernel, padding=1, groups=x.size(1))
    return filtered.abs().mean()


__all__ = ["AutoencoderBackboneConfig", "WatermarkAutoencoder", "FiLMConditioner", "frequency_regularizer"]
