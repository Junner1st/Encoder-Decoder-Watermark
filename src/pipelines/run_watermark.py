from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import yaml
except ImportError as exc:  # pragma: no cover - handled at runtime
    yaml = None

from src.models.msg_decoder import MessageDecoder
from src.models.watermark_autoencoder import AutoencoderBackboneConfig, WatermarkAutoencoder, frequency_regularizer
from src.utils.attacks import AttackParams, AttackScenario, apply_stage_attacks, build_eval_scenarios
from src.utils.ecc import ECCConfig, bit_accuracy, encode_bits, majority_decode
from src.utils.SSIM import compute_ssim


def _resolve_device(preference: str) -> torch.device:
    normalized = preference.strip().lower()
    if normalized in {"auto", "gpu"}:
        normalized = "cuda" if torch.cuda.is_available() else "cpu"

    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available. Install CUDA-enabled PyTorch or set device: cpu.")
        return torch.device(normalized)

    return torch.device(normalized)


def _next_exp_dir(runs_root: Path) -> Path:
    runs_root.mkdir(parents=True, exist_ok=True)
    max_id = 0
    for d in runs_root.iterdir():
        if d.is_dir() and d.name.startswith("exp"):
            try:
                num = int(d.name[3:])
                max_id = max(max_id, num)
            except ValueError:
                continue
    exp_dir = runs_root / f"exp{max_id + 1}"
    exp_dir.mkdir(parents=True, exist_ok=False)
    return exp_dir


def _load_config(path: Path) -> Tuple[argparse.Namespace, AttackParams, ECCConfig]:
    if yaml is None:
        raise ImportError("PyYAML is required to read YAML configs. Install with `pip install pyyaml`.")

    with path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    cfg_dict = {}
    for key, value in raw_cfg.items():
        if isinstance(key, str):
            cfg_dict[key.lower()] = value
        else:
            cfg_dict[key] = value

    defaults = {
        "data_root": "./data/celeba",
        "runs_dir": "./runs/watermark",
        "dataset": "celeba",  # options: celeba, cifar10
        "download_data": False,
        "image_size": 256,
        "batch_size": 16,
        "num_workers": 4,
        "total_epochs": 28,
        "stage_epochs": [
            {"stage": 0, "epochs": 2},
            {"stage": 1, "epochs": 6},
            {"stage": 2, "epochs": 10},
            {"stage": 3, "epochs": 10},
        ],
        "lr": 2e-4,
        "beta1": 0.5,
        "beta2": 0.999,
        "payload_bits": 96,
        "repetition": 2,
        "interleave": True,
        "lambda_img": 5.0,
        "lambda_msg": 1.0,
        "lambda_reg": 0.0,
        "device": "cuda",
        "attack": {},
        "max_eval_batches": 50,
        "compute_ssim": True,
        "compute_lpips": False,
    }

    merged = defaults.copy()

    section_fields = {
        "data": ["image_size", "batch_size", "num_workers"],
        "training": ["total_epochs", "stage_epochs"],
        "optimizer": ["lr", "beta1", "beta2"],
        "payload": ["payload_bits", "repetition", "interleave"],
        "loss": ["lambda_img", "lambda_msg", "lambda_reg"],
        "evaluation": ["max_eval_batches", "compute_ssim", "compute_lpips"],
    }

    for key, value in cfg_dict.items():
        if key in section_fields:
            continue
        merged[key] = value

    for section, fields in section_fields.items():
        section_cfg = cfg_dict.get(section)
        if isinstance(section_cfg, dict):
            for field in fields:
                if field in section_cfg:
                    merged[field] = section_cfg[field]

    attack_defaults = {
        "jpeg_qualities": [95, 90, 80, 70, 60, 50],
        "resize_min": 0.5,
        "resize_max": 1.5,
        "crop_min": 0.05,
        "crop_max": 0.3,
        "rotate_degrees": [2.0, 5.0, 10.0],
        "blur_sigma_min": 0.3,
        "blur_sigma_max": 1.5,
        "noise_sigma_min": 0.004,
        "noise_sigma_max": 0.02,
        "color_jitter": {"brightness": 0.05, "contrast": 0.05, "saturation": 0.05, "hue": 0.02},
        "recompress": True,
    }
    attack_cfg_dict = {**attack_defaults, **(merged.get("attack") or {})}
    attack_params = AttackParams(**attack_cfg_dict)

    ecc_cfg = ECCConfig(
        payload_bits=merged["payload_bits"],
        repetition=merged.get("repetition", 2),
        interleave=bool(merged.get("interleave", True)),
    )

    merged["data_root"] = Path(merged["data_root"]).expanduser()
    merged["runs_dir"] = Path(merged["runs_dir"]).expanduser()

    return argparse.Namespace(**merged), attack_params, ecc_cfg


def _build_dataloaders(cfg: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    common_tf = transforms.Compose(
        [
            transforms.Resize(cfg.image_size),
            transforms.CenterCrop(cfg.image_size),
            transforms.ToTensor(),
        ]
    )

    dataset_name = cfg.dataset.lower()
    download_flag = bool(cfg.download_data)

    if dataset_name == "celeba":
        train_ds = datasets.CelebA(
            root=str(cfg.data_root),
            split="train",
            transform=common_tf,
            target_type="identity",
            download=download_flag,
        )
        val_ds = datasets.CelebA(
            root=str(cfg.data_root),
            split="valid",
            transform=common_tf,
            target_type="identity",
            download=download_flag,
        )
    elif dataset_name == "cifar10":
        train_ds = datasets.CIFAR10(
            root=str(cfg.data_root),
            train=True,
            transform=common_tf,
            download=download_flag,
        )
        val_ds = datasets.CIFAR10(
            root=str(cfg.data_root),
            train=False,
            transform=common_tf,
            download=download_flag,
        )
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}. Use 'celeba' or 'cifar10'.")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def _stage_for_epoch(epoch: int, schedule: List[Dict]) -> int:
    remaining = epoch
    for item in schedule:
        st = int(item.get("stage", 0))
        remaining -= int(item.get("epochs", 0))
        if remaining <= 0:
            return st
    return int(schedule[-1].get("stage", 0))


def _accuracies_from_logits(logits: torch.Tensor, code: torch.Tensor, ecc: ECCConfig) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    hard_code = (probs >= 0.5).float()
    decoded_payload = majority_decode(hard_code, ecc)
    payload = majority_decode(code, ecc)
    code_acc = bit_accuracy(hard_code, code)
    payload_acc = bit_accuracy(decoded_payload, payload)
    payload_success = float((decoded_payload == payload).float().prod(dim=1).mean().item())
    return {
        "code_acc": code_acc,
        "payload_acc": payload_acc,
        "payload_success": payload_success,
    }


def decode_payload_from_code(code: torch.Tensor, ecc: ECCConfig) -> torch.Tensor:
    # Reverse repetition/interleave
    if code.size(1) != ecc.payload_bits * ecc.repetition:
        raise ValueError("Code length mismatch")
    cw = code
    if ecc.interleave:
        g = torch.Generator(device=cw.device)
        g.manual_seed(0)
        perm = torch.randperm(cw.size(1), device=cw.device, generator=g)
        inv_perm = torch.argsort(perm)
        cw = cw[:, inv_perm]
    grouped = cw.view(cw.size(0), ecc.payload_bits, ecc.repetition)
    votes = grouped.mean(dim=2)
    return (votes >= 0.5).float()


def psnr(recon: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((recon - target) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def _compute_ssim_batch(recon: torch.Tensor, target: torch.Tensor) -> float:
    scores = []
    for idx in range(recon.size(0)):
        a = target[idx].detach().cpu().permute(1, 2, 0).numpy()
        b = recon[idx].detach().cpu().permute(1, 2, 0).numpy()
        scores.append(float(compute_ssim(a, b)))
    return float(sum(scores) / max(len(scores), 1))


def train_epoch(
    model: WatermarkAutoencoder,
    msg_decoder: MessageDecoder,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    ecc: ECCConfig,
    attack_params: AttackParams,
    stage: int,
    cfg: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    msg_decoder.train()
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    total_loss = total_img = total_msg = 0.0
    total_batches = 0

    for batch in loader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(device, non_blocking=True)

        payload = torch.randint(0, 2, (images.size(0), ecc.payload_bits), device=device).float()
        code = encode_bits(payload, ecc)

        optimizer.zero_grad(set_to_none=True)

        recon = model(images, code)
        attacked = apply_stage_attacks(recon, attack_params, size=cfg.image_size, stage=stage, differentiable=True)
        logits, _ = msg_decoder(attacked)

        msg_loss = bce(logits, code)
        img_loss = l1(recon, images)
        reg_loss = frequency_regularizer(recon) if cfg.lambda_reg > 0 else recon.new_tensor(0.0)

        loss = cfg.lambda_img * img_loss + cfg.lambda_msg * msg_loss + cfg.lambda_reg * reg_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_img += img_loss.item()
        total_msg += msg_loss.item()
        total_batches += 1

    return {
        "loss": total_loss / max(total_batches, 1),
        "img_loss": total_img / max(total_batches, 1),
        "msg_loss": total_msg / max(total_batches, 1),
    }


def evaluate(
    model: WatermarkAutoencoder,
    msg_decoder: MessageDecoder,
    loader: DataLoader,
    device: torch.device,
    ecc: ECCConfig,
    scenarios: List,
    cfg: argparse.Namespace,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    msg_decoder.eval()
    bce = nn.BCEWithLogitsLoss(reduction="none")

    results: Dict[str, Dict[str, float]] = {}
    for scenario in scenarios:
        name = scenario.name
        code_bits_total = 0
        code_bit_errors = 0
        payload_success = 0
        payload_total = 0
        psnr_scores: List[float] = []
        ssim_scores: List[float] = []

        with torch.no_grad():
            for idx, batch in enumerate(loader):
                if idx >= cfg.max_eval_batches:
                    break
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                images = images.to(device, non_blocking=True)

                payload = torch.randint(0, 2, (images.size(0), ecc.payload_bits), device=device).float()
                code = encode_bits(payload, ecc)

                recon = model(images, code)
                attacked = scenario.fn(recon)
                logits, _ = msg_decoder(attacked)
                probs = torch.sigmoid(logits)
                hard = (probs >= 0.5).float()
                payload_pred = majority_decode(hard, ecc)
                payload_match = (payload_pred == payload).float().prod(dim=1)

                code_bits_total += code.numel()
                code_bit_errors += (hard != code).float().sum().item()
                payload_success += payload_match.sum().item()
                payload_total += payload_match.numel()

                psnr_scores.append(psnr(recon, images))
                if cfg.compute_ssim:
                    ssim_scores.append(_compute_ssim_batch(recon, images))

        code_ber = code_bit_errors / max(1, code_bits_total)
        payload_sr = payload_success / max(1, payload_total)
        results[name] = {
            "code_BER": code_ber,
            "payload_success": payload_sr,
            "psnr": float(sum(psnr_scores) / max(len(psnr_scores), 1)),
            "ssim": float(sum(ssim_scores) / max(len(ssim_scores), 1)) if ssim_scores else float("nan"),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Watermark PoC training (FiLM-conditioned AE)")
    parser.add_argument("--config", type=Path, default=Path("watermark_config.yml"), help="Path to YAML config")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run evaluation")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to checkpoint to load")
    args = parser.parse_args()

    cfg, attack_params, ecc_cfg = _load_config(args.config)
    device = _resolve_device(cfg.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using CUDA device: {torch.cuda.get_device_name(device.index or 0)}")
    else:
        print(f"Using device: {device}")

    train_loader, val_loader = _build_dataloaders(cfg)
    print(f"Using dataset: {cfg.dataset} | data_root: {cfg.data_root}")

    code_bits = ecc_cfg.payload_bits * ecc_cfg.repetition
    backbone_cfg = AutoencoderBackboneConfig(in_channels=3, base_channels=64, num_scales=4)
    model = WatermarkAutoencoder(backbone_cfg, code_bits=code_bits).to(device)
    msg_decoder = MessageDecoder(code_bits=code_bits).to(device)

    optimizer = Adam(
        list(model.parameters()) + list(msg_decoder.parameters()),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
    )

    exp_dir = _next_exp_dir(cfg.runs_dir)
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    with (exp_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(cfg).items()}, f, indent=2)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        msg_decoder.load_state_dict(ckpt["msg_decoder"])
        optimizer.load_state_dict(ckpt.get("optimizer", optimizer.state_dict()))
        print(f"Loaded checkpoint from {args.checkpoint}")

    scenarios: List[AttackScenario] = [AttackScenario(name="clean", fn=lambda x: x)]
    scenarios += build_eval_scenarios(attack_params, size=cfg.image_size)

    if args.eval_only:
        metrics = evaluate(model, msg_decoder, val_loader, device, ecc_cfg, scenarios, cfg)
        print(json.dumps(metrics, indent=2))
        return

    stage_schedule = cfg.stage_epochs
    total_epochs = cfg.total_epochs
    history: List[Dict] = []
    best_payload = 0.0

    for epoch in range(1, total_epochs + 1):
        stage = _stage_for_epoch(epoch, stage_schedule)
        train_metrics = train_epoch(
            model,
            msg_decoder,
            optimizer,
            train_loader,
            device,
            ecc_cfg,
            attack_params,
            stage,
            cfg,
        )

        eval_metrics = evaluate(model, msg_decoder, val_loader, device, ecc_cfg, scenarios, cfg)
        clean_payload = eval_metrics.get("clean", {}).get("payload_success", 0.0)
        if clean_payload > best_payload:
            best_payload = clean_payload
            torch.save(
                {
                    "model": model.state_dict(),
                    "msg_decoder": msg_decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                exp_dir / "checkpoints" / "best.pt",
            )

        torch.save(
            {
                "model": model.state_dict(),
                "msg_decoder": msg_decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            exp_dir / "checkpoints" / "last.pt",
        )

        record = {
            "epoch": epoch,
            "stage": stage,
            "train": train_metrics,
            "eval": eval_metrics,
        }
        history.append(record)
        with (exp_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(
            f"Epoch {epoch:03d} stage={stage} | "
            f"train_loss={train_metrics['loss']:.4f} img={train_metrics['img_loss']:.4f} msg={train_metrics['msg_loss']:.4f} | "
            f"clean_payload_sr={clean_payload:.3f}"
        )

    print("Training finished. Best clean payload success:", best_payload)


if __name__ == "__main__":
    main()
