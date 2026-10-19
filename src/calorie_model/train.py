"""Train the nutrition regressor, and optionally a food / not-food output.

    python -m calorie_model.prepare_data --out data/nutrition5k
    python -m calorie_model.train --data-root data/nutrition5k --epochs 30

    # With the food / not-food output, so a photo of a dog is not given calories:
    python -m calorie_model.prepare_openimages --out data/openimages
    python -m calorie_model.train --data-root data/nutrition5k --openimages-root data/openimages

    python -m calorie_model.train --synthetic --epochs 2 --no-pretrained   # smoke test, no data needed

All images are decoded once into memory on the training device and augmented there
(see ``batches.py``), so there are no DataLoader worker processes.

Only photos with nutrition labels train the calorie/macro outputs; every photo trains the food output.
The best epoch is chosen on a validation split carved out of the training data. The official
Nutrition5k test split is evaluated once, at the end, next to a trivial baseline (always predict the
training median); the food output is checked on held-out Open Images photos it never saw.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from .batches import Batches, load_bank
from .data import (
    FOOD_COL,
    KNOWN_START,
    N_OUT,
    WEIGHT_COL,
    ExtraNutritionDataset,
    Nutrition5kDataset,
    OpenImagesDataset,
    SyntheticDataset,
)
from .model import TARGETS, NutritionNet, count_parameters, scale_vector

# Calories is the headline number, so it gets the largest loss weight.
LOSS_WEIGHTS = {"calories": 2.0, "mass": 1.0, "fat": 1.0, "carb": 1.0, "protein": 1.0}
THRESHOLDS = (0.2, 0.35, 0.5, 0.65, 0.8)


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _metrics(abs_err_sum: torch.Tensor, true_sum: torch.Tensor, counts: torch.Tensor) -> dict[str, float]:
    """Per-output MAE over the photos that have that label; outputs with no labels are left out."""
    metrics = {f"mae_{t}": float(abs_err_sum[i] / counts[i]) for i, t in enumerate(TARGETS) if counts[i] > 0}
    if counts[0] > 0:
        metrics["mae_calories_pct"] = 100.0 * float(abs_err_sum[0] / true_sum[0].clamp(min=1e-6))
    return metrics


def roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Area under the ROC curve via ranks (ties are ranked arbitrarily, which is fine at this scale)."""
    pos = labels > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = scores.argsort().argsort().float() + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def gate_metrics(probs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    """Food / not-food quality: accuracy, AUC, share of food accepted, share of non-food rejected."""
    accepted, food = probs >= threshold, labels > 0.5
    out = {"gate_n": float(len(labels)), "gate_acc": float((accepted == food).float().mean()), "gate_auc": roc_auc(probs, labels)}
    if food.any():
        out["food_recall"] = float(accepted[food].float().mean())
    if (~food).any():
        out["nonfood_rejected"] = float((~accepted[~food]).float().mean())
    return out


@torch.no_grad()
def predict_food(model: nn.Module, batches: Iterable[tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Food probability and true food label (CPU tensors) for every image in ``batches``."""
    model.eval()
    probs, labels = [], []
    for x, y in batches:
        probs.append(torch.sigmoid(model(x.to(device))[:, FOOD_COL]).cpu())
        labels.append(y[:, FOOD_COL].cpu())
    return torch.cat(probs), torch.cat(labels)


@torch.no_grad()
def evaluate(model: nn.Module, batches: Iterable[tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> dict[str, float]:
    """Nutrition MAE in real units over the photos that have nutrition labels (plus calorie MAE as a
    percent of mean true calories) and, if the model has a food output, food / not-food metrics over all photos."""
    model.eval()
    scale = scale_vector(device)
    abs_err = torch.zeros(N_OUT, device=device)
    true_sum = torch.zeros(N_OUT, device=device)
    counts = torch.zeros(N_OUT, device=device)
    probs, labels = [], []
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        known = y[:, KNOWN_START : KNOWN_START + N_OUT]  # which labels are real, per photo and output
        abs_err += ((pred[:, :N_OUT] - y[:, :N_OUT]).abs() * scale * known).sum(0)
        true_sum += (y[:, :N_OUT] * scale * known).sum(0)
        counts += known.sum(0)
        if pred.shape[1] > FOOD_COL:
            probs.append(torch.sigmoid(pred[:, FOOD_COL]).cpu())
            labels.append(y[:, FOOD_COL].cpu())
    metrics = _metrics(abs_err.cpu(), true_sum.cpu(), counts.cpu())
    if probs:
        metrics.update(gate_metrics(torch.cat(probs), torch.cat(labels)))
    return metrics


def baseline_metrics(train_targets: torch.Tensor, test_targets: torch.Tensor) -> dict[str, float]:
    """Same metrics for a model that always predicts the per-target training median."""
    scale = scale_vector()
    train_known = train_targets[:, KNOWN_START : KNOWN_START + N_OUT] > 0
    median = torch.stack([train_targets[train_known[:, i], i].median() for i in range(N_OUT)])
    known = test_targets[:, KNOWN_START : KNOWN_START + N_OUT]
    err = (test_targets[:, :N_OUT] - median).abs() * scale * known
    return _metrics(err.sum(0), (test_targets[:, :N_OUT] * scale * known).sum(0), known.sum(0))


def compute_loss(pred: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, gate_weight: float) -> torch.Tensor:
    """Nutrition loss over the labels each photo really has, each photo scaled by how much its labels are
    trusted, plus the food / not-food loss over every photo."""
    per_target = F.smooth_l1_loss(pred[:, :N_OUT], y[:, :N_OUT], reduction="none", beta=0.1) * weights
    known = y[:, KNOWN_START : KNOWN_START + N_OUT]
    trust = y[:, WEIGHT_COL : WEIGHT_COL + 1]
    loss = (per_target * known * trust).sum() / known.sum().clamp(min=1.0)
    if pred.shape[1] > FOOD_COL:
        loss = loss + gate_weight * F.binary_cross_entropy_with_logits(pred[:, FOOD_COL], y[:, FOOD_COL])
    return loss


@dataclass
class Splits:
    train: list[Dataset]
    val: list[Dataset]
    test: Dataset  # nutrition test set (Nutrition5k)
    gate_test: OpenImagesDataset | None = field(default=None)  # held-out food / not-food photos
    extra_test: list[tuple[str, Dataset]] = field(default_factory=list)  # held-out photos of each extra source


def make_splits(args: argparse.Namespace) -> Splits:
    if args.synthetic:
        size = args.img_size
        return Splits([SyntheticDataset(64, size, 0)], [SyntheticDataset(32, size, 1)], SyntheticDataset(32, size, 2))
    kw = dict(img_size=args.img_size, views=tuple(args.views.split(",")), val_fraction=args.val_fraction)
    splits = Splits(
        train=[Nutrition5kDataset(args.data_root, "train", train_fraction=args.train_fraction, **kw)],
        val=[Nutrition5kDataset(args.data_root, "val", **kw)],
        test=Nutrition5kDataset(args.data_root, "test", **kw),
    )
    if args.openimages_root:
        oi = dict(img_size=args.img_size, val_fraction=args.val_fraction)
        splits.train.append(OpenImagesDataset(args.openimages_root, "train", **oi))
        splits.val.append(OpenImagesDataset(args.openimages_root, "val", **oi))
        splits.gate_test = OpenImagesDataset(args.openimages_root, "test", img_size=args.img_size)
    for root in args.extra_data:
        # Extra sources only add training data: the checkpoint is still chosen on clean Nutrition5k validation.
        splits.train.append(ExtraNutritionDataset(root, "train", args.img_size))
        try:
            splits.extra_test.append((root.name, ExtraNutritionDataset(root, "test", args.img_size)))
        except RuntimeError:
            pass  # this source has no held-out photos
    return splits


def make_batches(ds: Dataset | list[Dataset], args: argparse.Namespace, device: torch.device, train: bool = False) -> Batches:
    images, targets = load_bank(ds, device, args.load_threads)
    return Batches(images, targets, args.batch_size, shuffle=train, augment_=train)


def fmt(m: dict[str, float]) -> str:
    text = f"cal MAE {m['mae_calories']:.1f} kcal ({m['mae_calories_pct']:.1f}%)  mass MAE {m['mae_mass']:.1f} g"
    if "gate_acc" in m:
        text += f"  | food/not-food acc {100 * m['gate_acc']:.1f}%"
    return text


def fmt_all(m: dict[str, float]) -> str:
    """One line with every labelled output: calories (and percent error) plus mass and the three macros."""
    parts = []
    if "mae_calories" in m:
        parts.append(f"cal {m['mae_calories']:.1f} kcal" + (f" ({m['mae_calories_pct']:.1f}%)" if "mae_calories_pct" in m else ""))
    parts += [f"{t} {m[f'mae_{t}']:.1f} g" for t in TARGETS[1:] if f"mae_{t}" in m]
    return "  ".join(parts) or "(no labelled outputs)"


def pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}%"


def gate_report(model: nn.Module, splits: Splits, test_batches: Batches, args: argparse.Namespace, device: torch.device) -> dict:
    """Food / not-food quality on held-out Open Images photos, by group and at several thresholds."""
    assert splits.gate_test is not None
    probs, labels = predict_food(model, make_batches(splits.gate_test, args, device), device)
    groups = splits.gate_test.groups
    report: dict = {"images": len(labels), "overall": gate_metrics(probs, labels), "sweep": {}, "groups": {}}
    for t in THRESHOLDS:
        m = gate_metrics(probs, labels, t)
        report["sweep"][f"{t:.2f}"] = {k: m[k] for k in ("food_recall", "nonfood_rejected") if k in m}
    for g in sorted(set(groups)):
        sel = torch.tensor([x == g for x in groups])
        accepted = (probs[sel] >= 0.5).float().mean().item()
        report["groups"][g] = {"images": int(sel.sum()), "accepted_as_food": accepted}
    rig_probs, _ = predict_food(model, test_batches, device)  # Nutrition5k test photos are all food
    report["nutrition5k_test_recognised_as_food"] = float((rig_probs >= 0.5).float().mean())
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("data/nutrition5k"))
    p.add_argument("--openimages-root", type=Path, default=None, help="add the food / not-food output, trained on this Open Images cache")
    p.add_argument("--gate-weight", type=float, default=1.0, help="weight of the food / not-food loss")
    p.add_argument("--extra-data", type=Path, action="append", default=[], metavar="DIR",
                   help="extra food photos with (partial, maybe noisy) nutrition labels, see docs/adding-data.md; repeatable")
    p.add_argument("--views", default="overhead,side", help="comma-separated: overhead, side")
    p.add_argument("--val-fraction", type=float, default=0.1, help="share of training data held out for validation")
    p.add_argument("--synthetic", action="store_true", help="use procedural data (pipeline smoke test)")
    p.add_argument("--backbone", default="mobilenet_v3_small")
    p.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (no download)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--load-threads", type=int, default=min(8, os.cpu_count() or 1), help="threads used to decode images at startup")
    p.add_argument("--amp", action="store_true", help="mixed precision on CUDA (no speedup was measured on an RTX 3060)")
    p.add_argument("--train-fraction", type=float, default=1.0, help="use only this share of the training dishes (learning curves)")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=Path("runs/default"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    splits = make_splits(args)
    start = time.time()
    train_batches = make_batches(splits.train, args, device, train=True)
    val_batches, test_batches = make_batches(splits.val, args, device), make_batches(splits.test, args, device)
    n_train = sum(len(d) for d in splits.train)
    print(
        f"images: {n_train} train, {sum(len(d) for d in splits.val)} val, {len(splits.test)} test"
        + (f", {len(splits.gate_test)} held-out food/not-food" if splits.gate_test else "")
        + f" (decoded into {device} memory in {time.time() - start:.0f}s)"
    )

    food_head = splits.gate_test is not None
    model = NutritionNet(args.backbone, pretrained=not args.no_pretrained, food_head=food_head).to(device)
    use_amp = device.type == "cuda" and args.amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # input sizes are fixed, so autotuning pays off
    print(f"{args.backbone}: {count_parameters(model) / 1e6:.2f}M params, device={device}, amp={use_amp}, food output={food_head}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_batches)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
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
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(x)
            loss = compute_loss(pred.float(), y, weights, args.gate_weight)
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
                    "food_head": food_head,
                    "metrics": metrics,
                },
                args.out / "best.pt",
            )
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val calorie MAE {best:.1f} kcal -> {args.out / 'best.pt'}")

    # Final, one-off test-set report with the best checkpoint.
    model.load_state_dict(torch.load(args.out / "best.pt", map_location=device, weights_only=True)["state_dict"])
    report: dict = {
        "all": evaluate(model, test_batches, device),
        "baseline_median": baseline_metrics(splits.train[0].targets(), splits.test.targets()),  # Nutrition5k train
    }
    for name, ds in splits.extra_test:
        report[f"extra_{name}"] = evaluate(model, make_batches(ds, args, device), device)
        report[f"extra_{name}"]["images"] = len(ds)
    if not args.synthetic:
        for view in args.views.split(","):
            try:
                view_ds = Nutrition5kDataset(args.data_root, "test", args.img_size, views=(view,))
            except RuntimeError:
                continue
            report[f"view_{view}"] = evaluate(model, make_batches(view_ds, args, device), device)
            report[f"view_{view}"]["images"] = len(view_ds)
    gate = gate_report(model, splits, test_batches, args, device) if food_head else None
    if gate:
        report["gate"] = gate
    (args.out / "test_metrics.json").write_text(json.dumps(report, indent=2))

    print(f"\nTEST ({len(splits.test)} images)")
    for name, m in report.items():
        if name != "gate":
            print(f"  {name:16s} {fmt_all(m)}")
    if gate:
        o = gate["overall"]
        print(f"\nFOOD / NOT-FOOD on {gate['images']} held-out Open Images photos (accepted as food if probability >= 0.5)")
        print(f"  food accepted {pct(o.get('food_recall', float('nan')))}   non-food rejected {pct(o.get('nonfood_rejected', float('nan')))}   AUC {o['gate_auc']:.3f}")
        for g, v in gate["groups"].items():
            print(f"  {g:8s} {v['images']:5d} photos, accepted as food: {pct(v['accepted_as_food'])}")
        print(f"  Nutrition5k test photos recognised as food: {pct(gate['nutrition5k_test_recognised_as_food'])}")
        print("  threshold sweep (food accepted / non-food rejected): " + "  ".join(
            f"{t}: {pct(v.get('food_recall', float('nan')))}/{pct(v.get('nonfood_rejected', float('nan')))}" for t, v in gate["sweep"].items()))


if __name__ == "__main__":
    main()
