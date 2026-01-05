from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.models.autoencoder import build_celeb_a_autoencoder, celeb_a_transforms
from src.utils.evaluation import evaluate_autoencoder
from torchvision.utils import make_grid, save_image

try:
	import yaml
except ImportError as exc:  # pragma: no cover - handled at runtime
	yaml = None



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


def _build_dataloaders(
	data_root: Path,
	image_height: int,
	image_width: int,
	batch_size: int,
	num_workers: int,
	download: bool,
	pin_memory: bool,
) -> Tuple[DataLoader, DataLoader]:
	try:
		from torchvision import datasets
	except ImportError as exc:  # pragma: no cover - handled at runtime
		raise ImportError("torchvision is required for loading CelebA.") from exc

	transform = celeb_a_transforms(image_height=image_height, image_width=image_width)

	if not download and not data_root.exists():
		raise FileNotFoundError(
			f"CelebA not found at {data_root}. Set download: true in config.yml to download automatically."
		)

	train_ds = datasets.CelebA(
		root=str(data_root),
		split="train",
		transform=transform,
		target_type="identity",
		download=download,
	)
	val_ds = datasets.CelebA(
		root=str(data_root),
		split="valid",
		transform=transform,
		target_type="identity",
		download=download,
	)

	train_loader = DataLoader(
		train_ds,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)
	return train_loader, val_loader


def _train_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	optimizer: Adam,
	device: torch.device,
	max_steps: int | None = None,
) -> float:
	model.train()
	total_loss = 0.0
	total_items = 0

	for step, batch in enumerate(loader):
		if max_steps is not None and step >= max_steps:
			break

		images = batch[0] if isinstance(batch, (list, tuple)) else batch
		images = images.to(device, non_blocking=True)

		optimizer.zero_grad(set_to_none=True)
		recon = model(images)
		loss = F.mse_loss(recon, images, reduction="mean")
		loss.backward()
		optimizer.step()

		batch_items = images.size(0)
		total_loss += loss.item() * batch_items
		total_items += batch_items

	return total_loss / max(total_items, 1)


def _save_config(args: argparse.Namespace, exp_dir: Path) -> None:
	cfg_path = exp_dir / "config.json"
	serializable = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
	with cfg_path.open("w", encoding="utf-8") as f:
		json.dump(serializable, f, indent=2)


def _append_metrics(exp_dir: Path, metrics: Dict) -> None:
	metrics_path = exp_dir / "metrics.jsonl"
	with metrics_path.open("a", encoding="utf-8") as f:
		f.write(json.dumps(metrics) + "\n")


def _save_recon_grid(model: nn.Module, loader: DataLoader, device: torch.device, exp_dir: Path, epoch: int) -> None:
	model.eval()
	with torch.no_grad():
		batch = next(iter(loader))
		images = batch[0] if isinstance(batch, (list, tuple)) else batch
		images = images.to(device)
		recon = model(images)

		# Take first 8 samples
		images = images[:8].clamp(0, 1)
		recon = recon[:8].clamp(0, 1)

		grid_input = make_grid(images, nrow=4)
		grid_recon = make_grid(recon, nrow=4)
		stack = torch.cat([grid_input, grid_recon], dim=1)  # stack vertically
		out_path = exp_dir / f"recon_epoch{epoch:03d}.png"
		save_image(stack, out_path)


def _save_loss_plot(exp_dir: Path, history: list[Dict]) -> None:
	try:
		import matplotlib.pyplot as plt
	except ImportError:
		return

	epochs = [m["epoch"] for m in history]
	train = [m.get("train_loss", float("nan")) for m in history]
	val = [m.get("mse", float("nan")) for m in history]

	plt.figure(figsize=(6, 4))
	plt.plot(epochs, train, label="train_loss (MSE)")
	plt.plot(epochs, val, label="val_mse")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend()
	plt.tight_layout()
	plt.savefig(exp_dir / "loss_curve.png")
	plt.close()

def _load_config(config_path: Path) -> argparse.Namespace:
	if yaml is None:
		raise ImportError("PyYAML is required to read config.yml. Install with `pip install pyyaml`.")

	with config_path.open("r", encoding="utf-8") as f:
		cfg_dict = yaml.safe_load(f) or {}

	defaults = {
		"data_root": None,
		"runs_dir": "runs",
		"batch_size": 64,
		"epochs": 10,
		"lr": 2e-4,
		"beta1": 0.5,
		"beta2": 0.999,
		"num_workers": 4,
		"latent_dim": 256,
		"base_channels": 64,
		"image_height": 218,
		"image_width": 178,
		"input_channels": 3,
		"device": "cuda",
		"max_steps_per_epoch": None,
		"eval_ssim": False,
		"eval_lpips": False,
		"max_eval_batches": None,
		"download_celebA": False,
	}

	merged = {**defaults, **cfg_dict}

	if merged["data_root"] is None:
		raise ValueError("config.yml must set data_root.")

	# Normalize paths
	merged["data_root"] = Path(merged["data_root"]).expanduser()
	merged["runs_dir"] = Path(merged["runs_dir"]).expanduser()

	return argparse.Namespace(**merged)


def _resolve_device(preference: str) -> torch.device:
	"""Return a torch.device honoring GPU requests when available."""

	normalized = preference.strip().lower()
	if normalized in {"auto", "gpu"}:
		normalized = "cuda" if torch.cuda.is_available() else "cpu"

	if normalized.startswith("cuda"):
		if not torch.cuda.is_available():
			raise RuntimeError(
				"CUDA was requested but is not available. Install CUDA-enabled PyTorch or set device: cpu."
			)
		return torch.device(normalized)

	device = torch.device(normalized)
	return device


def main() -> None:
	parser = argparse.ArgumentParser(description="Train CelebA autoencoder from config.yml")
	parser.add_argument("--config", type=Path, default=Path("config.yml"), help="Path to YAML config")
	cli_args = parser.parse_args()

	args = _load_config(cli_args.config)

	device = _resolve_device(args.device)
	if device.type == "cuda":
		torch.backends.cudnn.benchmark = True
		print(f"Using CUDA device: {torch.cuda.get_device_name(device.index or 0)}")
	else:
		print(f"Using device: {device}")

	exp_dir = _next_exp_dir(args.runs_dir)
	_save_config(args, exp_dir)

	train_loader, val_loader = _build_dataloaders(
		data_root=args.data_root,
		image_height=args.image_height,
		image_width=args.image_width,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		download=args.download_celebA,
		pin_memory=device.type == "cuda",
	)

	model = build_celeb_a_autoencoder(
		image_height=args.image_height,
		image_width=args.image_width,
		latent_dim=args.latent_dim,
		base_channels=args.base_channels,
		input_channels=args.input_channels,
	).to(device)

	optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

	best_val_mse = float("inf")
	history: list[Dict] = []

	for epoch in range(1, args.epochs + 1):
		train_loss = _train_one_epoch(
			model=model,
			loader=train_loader,
			optimizer=optimizer,
			device=device,
			max_steps=args.max_steps_per_epoch,
		)

		eval_metrics = evaluate_autoencoder(
			model=model,
			dataloader=val_loader,
			device=device,
			max_batches=args.max_eval_batches,
			compute_ssim_score=args.eval_ssim,
			compute_lpips_score=args.eval_lpips,
		)

		metrics = {
			"epoch": epoch,
			"train_loss": train_loss,
			**eval_metrics,
		}
		_append_metrics(exp_dir, metrics)
		history.append(metrics)
		_save_loss_plot(exp_dir, history)
		_save_recon_grid(model, val_loader, device, exp_dir, epoch)

		torch.save({"model": model.state_dict()}, exp_dir / "last.pt")
		if eval_metrics.get("mse", float("inf")) < best_val_mse:
			best_val_mse = eval_metrics["mse"]
			torch.save({"model": model.state_dict()}, exp_dir / "best.pt")

		print(
			f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
			f"val_mse={eval_metrics.get('mse', float('nan')):.6f} | "
			f"val_psnr={eval_metrics.get('psnr', float('nan')):.2f}"
		)


if __name__ == "__main__":
	main()
