"""Accuracy AND consistency of a checkpoint on the official Nutrition5k test split.

    python -m calorie_model.evaluate_checkpoint runs/gate/best.pt --data-root data/nutrition5k

Prints, for the plain model and for the test-time-augmentation version (the photo mirrored four ways
and averaged, see ``export --tta``):

* the mean error of each output over all test photos, and over overhead / side photos separately;
* **consistency**: the same dish is photographed from up to five angles, so it should get about the
  same estimate from each. For every dish with at least two test photos this reports how far apart the
  calorie estimates are (range = highest minus lowest; std). Smaller is steadier.

Consistency is not accuracy: a model that always answers 250 kcal is perfectly consistent. Read the two
together.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .batches import load_bank
from .data import FOOD_COL, KNOWN_START, N_OUT, Nutrition5kDataset
from .export import deploy_model, load_checkpoint
from .model import TARGETS, scale_vector
from .train import get_device


def view_spread(calories: np.ndarray, dishes: list[str]) -> dict[str, float]:
    """How much the estimates for one dish differ between its photos (dishes with 2+ photos only)."""
    groups: dict[str, list[float]] = defaultdict(list)
    for value, dish in zip(calories, dishes):
        groups[dish].append(float(value))
    multi = [np.array(v) for v in groups.values() if len(v) >= 2]
    if not multi:
        return {"dishes": 0.0, "mean_range": float("nan"), "mean_std": float("nan"), "median_std": float("nan")}
    stds = np.array([v.std() for v in multi])
    return {
        "dishes": float(len(multi)),
        "mean_range": float(np.mean([v.max() - v.min() for v in multi])),
        "mean_std": float(stds.mean()),
        "median_std": float(np.median(stds)),
    }


@torch.no_grad()
def predict_all(net, images: torch.Tensor, tta: bool, batch_size: int = 64) -> np.ndarray:
    """Deployed-model outputs (real units; plus the food probability if the model has it) for a uint8 NCHW bank."""
    model = deploy_model(net, tta).to(images.device)
    outs = [model(images[i : i + batch_size].permute(0, 2, 3, 1)).cpu() for i in range(0, len(images), batch_size)]
    return torch.cat(outs).numpy()


def mae(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    rows = np.ones(len(pred), bool) if mask is None else mask
    err = np.abs(pred[rows, :N_OUT] - truth[rows])
    out = {f"mae_{t}": float(err[:, i].mean()) for i, t in enumerate(TARGETS)}
    out["mae_calories_pct"] = 100.0 * float(err[:, 0].mean() / truth[rows, 0].mean())
    return out


def evaluate_one(net, images, truth, dishes, views, tta: bool) -> dict:
    pred = predict_all(net, images, tta)
    views = np.array(views)
    result = {"all": mae(pred, truth), "consistency": view_spread(pred[:, 0], dishes)}
    for v in sorted(set(views)):
        result[f"view_{v}"] = mae(pred, truth, views == v)
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--data-root", type=Path, default=Path("data/nutrition5k"))
    p.add_argument("--views", default="overhead,side")
    p.add_argument("--device", default="auto")
    p.add_argument("--json", type=Path, default=None, help="also write the numbers to this file")
    args = p.parse_args(argv)

    device = get_device(args.device)
    net, size = load_checkpoint(args.checkpoint)
    net = net.to(device)
    ds = Nutrition5kDataset(args.data_root, "test", size, views=tuple(args.views.split(",")))
    images, targets = load_bank(ds, device)
    known = targets[:, KNOWN_START : KNOWN_START + N_OUT].cpu().numpy() > 0
    assert known.all(), "the Nutrition5k test split should have every label"
    truth = (targets[:, :N_OUT] * scale_vector(device)).cpu().numpy()

    results = {name: evaluate_one(net, images, truth, ds.dishes, ds.views, tta) for name, tta in (("plain", False), ("tta", True))}
    print(f"{len(ds)} test photos of {len(set(ds.dishes))} dishes, {args.checkpoint}\n")
    print(f"{'':6s} {'cal MAE':>16s} {'mass':>7s} {'fat':>6s} {'carb':>6s} {'protein':>8s}   {'per dish, calories between its photos:':s}")
    print(f"{'':6s} {'':>16s} {'':>7s} {'':>6s} {'':>6s} {'':>8s}   {'mean range':>11s} {'mean std':>9s}")
    for name, r in results.items():
        a, c = r["all"], r["consistency"]
        print(
            f"{name:6s} {a['mae_calories']:7.1f} ({a['mae_calories_pct']:4.1f}%) {a['mae_mass']:7.1f} {a['mae_fat']:6.1f} "
            f"{a['mae_carb']:6.1f} {a['mae_protein']:8.1f}   {c['mean_range']:8.1f} kcal {c['mean_std']:6.1f} kcal"
        )
    for view in sorted(k for k in results["plain"] if k.startswith("view_")):
        print(f"  {view:14s} plain {results['plain'][view]['mae_calories']:6.1f} kcal   tta {results['tta'][view]['mae_calories']:6.1f} kcal")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
