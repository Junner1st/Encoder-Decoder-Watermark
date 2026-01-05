from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class _ConvBlock(nn.Module):
	"""Down-sampling block used inside the encoder."""

	def __init__(self, in_channels: int, out_channels: int, *, use_bn: bool = True) -> None:
		super().__init__()
		layers = [
			nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not use_bn),
			nn.LeakyReLU(0.2, inplace=True),
		]
		if use_bn:
			layers.insert(1, nn.BatchNorm2d(out_channels))
		self.block = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
		return self.block(x)


class _DeconvBlock(nn.Module):
	"""Up-sampling block used inside the decoder."""

	def __init__(self, in_channels: int, out_channels: int, *, use_bn: bool = True) -> None:
		super().__init__()
		layers = [
			nn.ConvTranspose2d(
				in_channels,
				out_channels,
				kernel_size=4,
				stride=2,
				padding=1,
				bias=not use_bn,
			),
			nn.ReLU(inplace=True),
		]
		if use_bn:
			layers.insert(1, nn.BatchNorm2d(out_channels))
		self.block = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
		return self.block(x)


@dataclass
class AutoencoderConfig:
	"""Configuration for the CelebA autoencoder."""

	image_height: int = 218
	image_width: int = 178
	latent_dim: int = 256
	base_channels: int = 64
	input_channels: int = 3


class Autoencoder(nn.Module):
	"""Simple convolutional autoencoder suited for aligned CelebA crops."""

	def __init__(self, cfg: AutoencoderConfig) -> None:
		super().__init__()

		self.cfg = cfg
		height = cfg.image_height
		width = cfg.image_width
		spatial_dims = []
		for _ in range(4):
			height = self._downsample_dim(height)
			width = self._downsample_dim(width)
			spatial_dims.append((height, width))
		spatial_height, spatial_width = spatial_dims[-1]

		encoder_channels = [cfg.base_channels, cfg.base_channels * 2, cfg.base_channels * 4, cfg.base_channels * 8]

		encoder_layers = []
		in_ch = cfg.input_channels
		for idx, out_ch in enumerate(encoder_channels):
			# The first block omits batch-norm following DCGAN best-practices.
			encoder_layers.append(_ConvBlock(in_ch, out_ch, use_bn=idx != 0))
			in_ch = out_ch
		self.encoder = nn.Sequential(*encoder_layers)

		latent_input_dim = encoder_channels[-1] * spatial_height * spatial_width
		self.to_latent = nn.Linear(latent_input_dim, cfg.latent_dim)
		self.from_latent = nn.Linear(cfg.latent_dim, latent_input_dim)

		decoder_layers = [
			_DeconvBlock(encoder_channels[-1], encoder_channels[-2]),
			_DeconvBlock(encoder_channels[-2], encoder_channels[-3]),
			_DeconvBlock(encoder_channels[-3], encoder_channels[-4]),
			_DeconvBlock(encoder_channels[-4], cfg.base_channels, use_bn=False),
			nn.ConvTranspose2d(cfg.base_channels, cfg.input_channels, kernel_size=4, stride=2, padding=1),
			nn.Sigmoid(),  # Output pixels in [0, 1]
		]
		self.decoder = nn.Sequential(*decoder_layers)

		self._spatial_shape = (spatial_height, spatial_width)
		self._channels = encoder_channels[-1]

		self.apply(self._init_weights)

	@staticmethod
	def _downsample_dim(dim: int) -> int:
		return (dim + 2 - 4) // 2 + 1

	@staticmethod
	def _init_weights(module: nn.Module) -> None:
		"""He initialization for convs and Xavier for linears."""

		if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
			nn.init.kaiming_normal_(module.weight)
			if module.bias is not None:
				nn.init.zeros_(module.bias)
		elif isinstance(module, nn.BatchNorm2d):
			nn.init.ones_(module.weight)
			nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Linear):
			nn.init.xavier_uniform_(module.weight)
			nn.init.zeros_(module.bias)

	def encode(self, x: torch.Tensor) -> torch.Tensor:
		"""Encode an input batch into latent vectors."""

		feats = self.encoder(x)
		flattened = feats.view(feats.size(0), -1)
		latent = self.to_latent(flattened)
		return latent

	def decode(self, latent: torch.Tensor) -> torch.Tensor:
		"""Decode latent vectors back into image space."""

		projected = self.from_latent(latent)
		reshaped = projected.view(latent.size(0), self._channels, *self._spatial_shape)
		recon = self.decoder(reshaped)
		recon = self._match_target_shape(recon)
		return recon

	def _match_target_shape(self, tensor: torch.Tensor) -> torch.Tensor:
		_, _, h, w = tensor.shape
		target_h = self.cfg.image_height
		target_w = self.cfg.image_width
		pad_top = pad_bottom = pad_left = pad_right = 0

		if h < target_h:
			delta = target_h - h
			pad_top = delta // 2
			pad_bottom = delta - pad_top
		elif h > target_h:
			crop_top = (h - target_h) // 2
			tensor = tensor[:, :, crop_top : crop_top + target_h, :]

		if w < target_w:
			delta = target_w - w
			pad_left = delta // 2
			pad_right = delta - pad_left
		elif w > target_w:
			crop_left = (w - target_w) // 2
			tensor = tensor[:, :, :, crop_left : crop_left + target_w]

		if any(v > 0 for v in (pad_left, pad_right, pad_top, pad_bottom)):
			tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
		return tensor

	def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
		latent = self.encode(x)
		recon = self.decode(latent)
		return recon


def build_celeb_a_autoencoder(
	*,
	image_height: int = 218,
	image_width: int = 178,
	latent_dim: int = 256,
	base_channels: int = 64,
	input_channels: int = 3,
) -> Autoencoder:
	"""Convenience constructor used by training scripts."""

	cfg = AutoencoderConfig(
		image_height=image_height,
		image_width=image_width,
		latent_dim=latent_dim,
		base_channels=base_channels,
		input_channels=input_channels,
	)
	return Autoencoder(cfg)


def celeb_a_transforms(
	*,
	image_height: int = 218,
	image_width: int = 178,
) -> "torchvision.transforms.Compose":
	"""Return default preprocessing for CelebA (lazy-imports torchvision)."""

	from torchvision import transforms  # Import lazily to avoid hard dependency at import time

	return transforms.Compose(
		[
			transforms.CenterCrop((image_height, image_width)),
			transforms.ToTensor(),
		]
	)

