# Autoencoder Watermark

This repo hosts two closely related pipelines:

- `src/pipelines/run_training.py` — the original CelebA autoencoder baseline.
- `src/pipelines/run_watermark.py` — a FiLM-conditioned U-Net that hides a short payload, survives differentiable attacks, and verifies the message with an auxiliary decoder head.

Both pipelines write experiment folders under `runs/` (watermark experiments live in `runs/watermark/exp*`).

## Environment

```bash
conda env create -f conda-environment.yml
conda activate autoencoder_310
```

Regenerate the env file after upgrades:

```bash
conda env export | sed '$d' > conda-environment.yml
```

### Extra Python deps

- `pip install opencv-python` (required by `src/utils/SSIM.py`).
- `pip install pyyaml` (already listed, needed for YAML configs).

## Data

The watermark pipeline supports CelebA and CIFAR-10 via torchvision.

| Dataset | Config `dataset` | Default root | Notes |
|---------|------------------|--------------|-------|
| CelebA  | `celeba`         | `./data/celeba` | Requires aligned splits. Set `download_data: true` to let torchvision fetch it. |
| CIFAR-10| `cifar10`        | `./data/cifar-10-python` | Matches the unpacked `cifar-10-batches-py` format. |

Set `download_data: true` the first time to pull files automatically; set it back to `false` when the dataset is already present to avoid repeated downloads.

## Running Experiments

### Baseline autoencoder

```bash
python -m src.pipelines.run_training --config config.yml
```

Metrics and checkpoints are stored under `runs/exp*/`.

### Watermark PoC

```bash
python -m src.pipelines.run_watermark --config watermark_config.yml
```

Key runtime flags:

- `--eval-only` — skip training and run evaluation/attack sweeps using the specified config and checkpoint.
- `--checkpoint path/to/ckpt.pt` — resume or evaluate a saved experiment.

Each run creates `runs/watermark/exp*/` containing `metrics.jsonl`, visualizations (recon grids, loss curves), and checkpoints under `checkpoints/`.

## `watermark_config.yml` overview

Top-level keys describe logical sections. Defaults are tuned for CIFAR-10; change as needed.

- `data_root`, `runs_dir`, `dataset`, `download_data`, `device` — global paths and hardware selection.
- `Data` — `image_size`, `batch_size`, `num_workers`.
- `Training` — `total_epochs` and curriculum `stage_epochs` (inline JSON style retained, e.g. `{stage: 2, epochs: 10}`).
- `Optimizer` — Adam hyperparameters.
- `Payload` — payload bit length, repetition/interleave ECC knobs.
- `Loss` — reconstruction vs. message loss weights and optional frequency regularizer.
- `Attack` — ranges for JPEG qualities, resize, crop, rotate, blur, noise, color jitter (kept inline to mirror JSON), and recompression toggle.
- `Evaluation` — number of batches and whether to compute SSIM/LPIPS.

The script logs which dataset/root is being used at startup for clarity.

## Evaluation & Metrics

During training the pipeline logs:

- Reconstruction metrics: PSNR, SSIM (optional), and LPIPS if enabled.
- Message metrics: bit error rate, payload accuracy, ECC success rate, confidence.
- Attack sweeps: clean + JPEG qualities, crop ratios, rotation angles, and a combined harsh pipeline.

`metrics.jsonl` stores per-epoch summaries for easy plotting (BER vs. quality, payload success vs. attack strength, etc.).

## Troubleshooting

- **Dataset missing** — ensure `data_root` points to the correct folder structure or set `download_data: true` to let torchvision populate it.
- **`cv2.filter2D` unavailable** — OpenCV-lite wheels sometimes omit this symbol; our SSIM utility now falls back to PyTorch convolution automatically.
- **CUDA not detected** — confirm the active Conda env uses a CUDA-enabled PyTorch wheel (`python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"`). Set `device: cpu` in the config to override.

## Tested Hardware

- CPU: AMD Ryzen 9 9950X
- GPU: NVIDIA RTX 4090 (CUDA 13.0 drivers)


