from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class ECCConfig:
    payload_bits: int = 96
    repetition: int = 2
    interleave: bool = True


def _interleave_perm(length: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(0)
    return torch.randperm(length, device=device, generator=g)


def encode_bits(payload: torch.Tensor, cfg: ECCConfig) -> torch.Tensor:
    """Simple repetition + optional interleaving ECC baseline.

    Args:
        payload: (B, payload_bits) binary tensor.
        cfg: ECCConfig
    Returns:
        codeword: (B, payload_bits * repetition) binary tensor.
    """

    if payload.dim() != 2:
        raise ValueError("payload must be (B, payload_bits)")
    if payload.size(1) != cfg.payload_bits:
        raise ValueError(f"Expected payload_bits={cfg.payload_bits}, got {payload.size(1)}")

    code = payload.unsqueeze(-1).expand(-1, -1, cfg.repetition).reshape(payload.size(0), -1)
    if cfg.interleave:
        perm = _interleave_perm(code.size(1), device=code.device)
        code = code[:, perm]
    return code


def decode_bits(codeword: torch.Tensor, cfg: ECCConfig) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Decode the repetition/interleaved codeword.

    Returns tuple: (decoded_bits, hard_codeword, vote_margin).
    """

    if codeword.dim() != 2:
        raise ValueError("codeword must be (B, L)")

    if codeword.size(1) != cfg.payload_bits * cfg.repetition:
        raise ValueError("codeword length does not match repetition scheme")

    cw = codeword
    if cfg.interleave:
        perm = _interleave_perm(cw.size(1), device=cw.device)
        inv_perm = torch.argsort(perm)
        cw = cw[:, inv_perm]

    grouped = cw.view(cw.size(0), cfg.payload_bits, cfg.repetition)
    votes = grouped.mean(dim=2)
    decoded = (votes >= 0.5).float()
    hard_bits = (cw >= 0.5).float()
    confidence = float((votes - 0.5).abs().mean().item())
    return decoded, hard_bits, confidence


def majority_decode(codeword: torch.Tensor, cfg: ECCConfig) -> torch.Tensor:
    decoded, _, _ = decode_bits(codeword, cfg)
    return decoded


def bit_accuracy(pred_bits: torch.Tensor, target_bits: torch.Tensor) -> float:
    return float((pred_bits == target_bits).float().mean().item())


__all__ = ["ECCConfig", "encode_bits", "decode_bits", "majority_decode", "bit_accuracy"]
