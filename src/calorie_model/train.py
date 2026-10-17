"""Train the nutrition regressor.

    python -m calorie_model.prepare_data --out data/nutrition5k
    python -m calorie_model.train --data-root data/nutrition5k --epochs 30
    python -m calorie_model.train --synthetic --epochs 2 --no-pretrained   # smoke test, no data needed

All images are decoded once into memory on the training device and augmented there
(see ``batches.py``), so there are no DataLoader worker processes.

The best epoch is chosen on a validation split carved out of the training dishes.
The official test split is evaluated once, at the end, next to a trivial baseline
(always predict the training median), so the numbers can be put in context.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from .batches import Batches, load_bank
from .data import Nutrition5kDataset, SyntheticDataset
from .model import TARGETS, NutritionNet, count_parameters, scale_vector

# Calories is the headline number, so it gets the largest loss weight.
LOSS_WEIGHTS = {"calories": 2.0, "mass": 1.0, "fat": 1.0, "carb": 1.0, "protein": 1.0}


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _metrics(abs_err_sum: torch.Tensor, true_sum: torch.Tensor, n: int) -> dict[str, float]:
    metrics = {f"mae_{t}": float(abs_err_sum[i] / n) for i, t in enumerate(TARGETS)}
    metrics["mae_calories_pct"] = 100.0 * float(abs_err_sum[0] / true_sum[0].clamp(min=1e-6))
    return metrics


@torch.no_grad()
def evaluate(model: nn.Module, batches: Iterable[tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> dict[str, float]:
    """Per-target MAE in real units, plus calorie MAE as a percent of mean true calories."""
    model.eval()
    scale = scale_vector(device)
    abs_err = torch.zeros(len(TARGETS), device=device)
    true_sum = torch.zeros(len(TARGETS), device=device)
    n = 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        abs_err += ((pred - y).abs() * scale).sum(0)
        true_sum += (y * scale).sum(0)
        n += x.shape[0]
    return _metrics(abs_err.cpu(), true_sum.cpu(), n)


def baseline_metrics(train_targets: torch.Tensor, test_targets: torch.Tensor) -> dict[str, float]:
    """Same metrics for a model that always predicts the per-target training median."""
    scale = scale_vector()
    median = train_targets.median(dim=0).values
    abs_err = ((test_targets - median).abs() * scale).sum(0)
    return _metrics(abs_err, (test_targets * scale).sum(0), len(test_targets))


def make_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset, Dataset]:
    if args.synthetic:
        size = args.img_size
        return SyntheticDataset(64, size, 0), SyntheticDataset(32, size, 1), SyntheticDataset(32, size, 2)
    kw = dict(img_size=args.img_size, views=tuple(args.views.split(",")), val_fraction=args.val_fraction)
    return tuple(Nutrition5kDataset(args.data_root, split, **kw) for split in ("train", "val", "test"))


def make_batches(ds: Dataset, args: argparse.Namespace, device: torch.device, train: bool = False) -> Batches:
    images, targets = load_bank(ds, device, args.load_threads)
    return Batches(images, targets, args.batch_size, shuffle=train, augment_=train)


def fmt(m: dict[str, float]) -> str:
    return f"cal MAE {m['mae_calories']:.1f} kcal ({m['mae_calories_pct']:.1f}%)  mass MAE {m['mae_mass']:.1f} g"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("data/nutrition5k"))
    p.add_argument("--views", default="overhead,side", help="comma-separated: overhead, side")
    p.add_argument("--val-fraction", type=float, default=0.1, help="share of training dishes held out for validation")
    p.add_argument("--synthetic", action="store_true", help="use procedural data (pipeline smoke test)")
    p.add_argument("--backbone", default="mobilenet_v3_small")
    p.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (no download)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--load-threads", type=int, default=min(8, os.cpu_count() or 1), help="threads used to decode images at startup")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision on CUDA")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=Path("runs/default"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    train_ds, val_ds, test_ds = make_datasets(args)
    start = time.time()
    train_batches = make_batches(train_ds, args, device, train=True)
    val_batches, test_batches = make_batches(val_ds, args, device), make_batches(test_ds, args, device)
    print(
        f"images: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test "
        f"(decoded into {device} memory in {time.time() - start:.0f}s)"
    )

    model = NutritionNet(args.backbone, pretrained=not args.no_pretrained).to(device)
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    print(f"{args.backbone}: {count_parameters(model) / 1e6:.2f}M params, device={device}, amp={use_amp}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_batches)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.SmoothL1Loss(reduction="none", beta=0.1)
    weights = torch.tensor([LOSS_WEIGHTS[t] for t in TARGETS], device=device)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running, steps = 0.0, 0
        for x, y in train_batches:
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(x)
            loss = (criterion(pred.float(), y) * weights).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            steps += 1
        metrics = evaluate(model, val_batches, device)
        metrics.update(epoch=epoch, train_loss=running / max(steps, 1))
        history.append(metrics)
        print(
            f"epoch {epoch:3d}  {time.time() - epoch_start:4.0f}s  loss {metrics['train_loss']:.4f}  val {fmt(metrics)}",
            flush=True,
        )
        if metrics["mae_calories"] < best:
            best = metrics["mae_calories"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "img_size": args.img_size,
                    "metrics": metrics,
                },
                args.out / "best.pt",
            )
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val calorie MAE {best:.1f} kcal -> {args.out / 'best.pt'}")

    # Final, one-off test-set report with the best checkpoint.
    model.load_state_dict(torch.load(args.out / "best.pt", map_location=device, weights_only=True)["state_dict"])
    report = {
        "all": evaluate(model, test_batches, device),
        "baseline_median": baseline_metrics(train_ds.targets(), test_ds.targets()),
    }
    if not args.synthetic:
        for view in args.views.split(","):
            try:
                view_ds = Nutrition5kDataset(args.data_root, "test", args.img_size, views=(view,))
            except RuntimeError:
                continue
            report[f"view_{view}"] = evaluate(model, make_batches(view_ds, args, device), device)
            report[f"view_{view}"]["images"] = len(view_ds)
    (args.out / "test_metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nTEST ({len(test_ds)} images)")
    for name, m in report.items():
        print(f"  {name:16s} {fmt(m)}")


if __name__ == "__main__":
    main()
