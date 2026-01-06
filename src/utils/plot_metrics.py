from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Record:
    epoch: int
    stage: Optional[int]
    train: Mapping[str, float]
    eval: Mapping[str, Mapping[str, float]]


def _safe_float(x: Any) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(x)
    except Exception:
        return float("nan")


def load_metrics_jsonl(path: Path) -> List[Record]:
    records: List[Record] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc

            epoch = int(obj.get("epoch"))
            stage_val = obj.get("stage")
            stage = int(stage_val) if stage_val is not None else None

            train = obj.get("train") or {}
            if not isinstance(train, dict):
                train = {}

            eval_dict = obj.get("eval") or {}
            if not isinstance(eval_dict, dict):
                eval_dict = {}

            # Ensure nested dicts
            normalized_eval: Dict[str, Dict[str, float]] = {}
            for scenario, metrics in eval_dict.items():
                if isinstance(metrics, dict):
                    normalized_eval[str(scenario)] = {str(k): _safe_float(v) for k, v in metrics.items()}

            records.append(
                Record(
                    epoch=epoch,
                    stage=stage,
                    train={str(k): _safe_float(v) for k, v in train.items()},
                    eval=normalized_eval,
                )
            )

    records.sort(key=lambda r: r.epoch)
    return records


def _ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def _plot_lines(
    *,
    x: Sequence[int],
    series: Sequence[Tuple[str, Sequence[float]]],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    stage_by_epoch: Optional[Mapping[int, int]] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting. Install with `pip install matplotlib`.") from exc

    plt.figure(figsize=(10, 5))

    for label, y in series:
        plt.plot(x, y, label=label, linewidth=1.5)

    if stage_by_epoch:
        # Mark stage boundaries with vertical dotted lines.
        prev_stage: Optional[int] = None
        for epoch in x:
            st = stage_by_epoch.get(epoch)
            if prev_stage is None:
                prev_stage = st
                continue
            if st != prev_stage:
                plt.axvline(epoch, color="gray", linestyle=":", linewidth=1)
                prev_stage = st

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

    if len(series) <= 6:
        plt.legend(loc="best")
    else:
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _collect_train_metrics(records: Sequence[Record]) -> List[str]:
    keys: set[str] = set()
    for r in records:
        keys.update(r.train.keys())
    # Keep common ordering first.
    preferred = ["loss", "img_loss", "msg_loss"]
    rest = sorted(k for k in keys if k not in preferred)
    return [k for k in preferred if k in keys] + rest


def _collect_eval_scenarios(records: Sequence[Record]) -> List[str]:
    scenarios: set[str] = set()
    for r in records:
        scenarios.update(r.eval.keys())
    # Put clean first if present.
    ordered: List[str] = []
    if "clean" in scenarios:
        ordered.append("clean")
    for s in sorted(scenarios):
        if s != "clean":
            ordered.append(s)
    return ordered


def _collect_eval_metrics(records: Sequence[Record]) -> List[str]:
    metric_keys: set[str] = set()
    for r in records:
        for metrics in r.eval.values():
            metric_keys.update(metrics.keys())

    preferred = ["payload_success", "payload_bit_acc", "payload_BER", "code_BER", "psnr", "ssim", "lpips"]
    rest = sorted(k for k in metric_keys if k not in preferred)
    return [k for k in preferred if k in metric_keys] + rest


def _nan_to_none_for_ylim(values: Iterable[float]) -> List[float]:
    out: List[float] = []
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append(float("nan"))
        else:
            out.append(float(v))
    return out


def plot_all(metrics_path: Path, outdir: Path) -> None:
    records = load_metrics_jsonl(metrics_path)
    if not records:
        raise ValueError(f"No records found in {metrics_path}")

    _ensure_outdir(outdir)

    epochs = [r.epoch for r in records]
    stage_by_epoch = {r.epoch: r.stage for r in records if r.stage is not None}

    # Train plots (one file per metric)
    train_keys = _collect_train_metrics(records)
    for k in train_keys:
        y = [_safe_float(r.train.get(k)) for r in records]
        _plot_lines(
            x=epochs,
            series=[(k, y)],
            title=f"Train {k} vs Epoch",
            xlabel="Epoch",
            ylabel=k,
            out_path=outdir / f"train_{k}.png",
            stage_by_epoch=stage_by_epoch,
        )

    # Eval plots: for each metric, draw one line per scenario.
    scenarios = _collect_eval_scenarios(records)
    eval_metrics = _collect_eval_metrics(records)

    for metric_name in eval_metrics:
        series: List[Tuple[str, List[float]]] = []
        for scenario in scenarios:
            y = []
            for r in records:
                v = r.eval.get(scenario, {}).get(metric_name)
                y.append(_safe_float(v))
            series.append((scenario, y))

        _plot_lines(
            x=epochs,
            series=series,
            title=f"Eval {metric_name} vs Epoch",
            xlabel="Epoch",
            ylabel=metric_name,
            out_path=outdir / f"eval_{metric_name}.png",
            stage_by_epoch=stage_by_epoch,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training/eval curves from watermark metrics.jsonl")
    parser.add_argument("metrics", type=Path, help="Path to metrics.jsonl")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for PNGs (default: <metrics_dir>/plots)",
    )
    args = parser.parse_args()

    metrics_path = args.metrics.expanduser().resolve()
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    outdir = args.outdir
    if outdir is None:
        outdir = metrics_path.parent / "plots"
    outdir = outdir.expanduser().resolve()

    plot_all(metrics_path, outdir)
    print(f"Wrote plots to: {outdir}")


if __name__ == "__main__":
    main()
