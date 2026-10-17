"""Train the nutrition regressor.

    python -m calorie_model.train --data-root data/nutrition5k --epochs 30
    python -m calorie_model.train --synthetic --epochs 2 --no-pretrained   # smoke test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Per-target MAE in real units, plus calorie MAE as a percent of mean true calories."""
    model.eval()
    scale = scale_vector(device)
    abs_err = torch.zeros(len(TARGETS), device=device)
    true_sum = torch.zeros(len(TARGETS), device=device)
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        abs_err += ((pred - y).abs() * scale).sum(0)
        true_sum += (y * scale).sum(0)
        n += x.shape[0]
    metrics = {f"mae_{t}": float(abs_err[i] / n) for i, t in enumerate(TARGETS)}
    metrics["mae_calories_pct"] = 100.0 * float(abs_err[0] / true_sum[0].clamp(min=1e-6))
    return metrics


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    if args.synthetic:
        train_ds = SyntheticDataset(n=64, img_size=args.img_size, seed=0)
        val_ds = SyntheticDataset(n=32, img_size=args.img_size, seed=1)
    else:
        train_ds = Nutrition5kDataset(args.data_root, "train", args.img_size, train=True)
        val_ds = Nutrition5kDataset(args.data_root, "test", args.img_size, train=False)
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    return DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) > args.batch_size, **kw), DataLoader(val_ds, **kw)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("data/nutrition5k"))
    p.add_argument("--synthetic", action="store_true", help="use procedural data (pipeline smoke test)")
    p.add_argument("--backbone", default="mobilenet_v3_small")
    p.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (no download)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=Path("runs/default"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    train_loader, val_loader = build_loaders(args)

    model = NutritionNet(args.backbone, pretrained=not args.no_pretrained).to(device)
    print(f"{args.backbone}: {count_parameters(model) / 1e6:.2f}M params, device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader)
    )
    criterion = nn.SmoothL1Loss(reduction="none", beta=0.1)
    weights = torch.tensor([LOSS_WEIGHTS[t] for t in TARGETS], device=device)

    args.out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, steps = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = (criterion(model(x), y) * weights).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
            steps += 1
        metrics = evaluate(model, val_loader, device)
        metrics.update(epoch=epoch, train_loss=running / max(steps, 1))
        history.append(metrics)
        print(
            f"epoch {epoch:3d}  loss {metrics['train_loss']:.4f}  "
            f"cal MAE {metrics['mae_calories']:.1f} kcal ({metrics['mae_calories_pct']:.1f}%)  "
            f"mass MAE {metrics['mae_mass']:.1f} g"
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
    print(f"best calorie MAE {best:.1f} kcal -> {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
