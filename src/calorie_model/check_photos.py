"""Run a trained checkpoint on your own photos, exactly as the app will, and print what it says.

    python -m calorie_model.check_photos runs/gate/best.pt dog.jpg lunch.jpg
    python -m calorie_model.check_photos runs/gate/best.pt phone_photos            # a whole folder
    python -m calorie_model.check_photos runs/gate/best.pt phone_photos --threshold 0.6

For a model with the food output, a photo whose food probability is below the threshold is reported as
NOT FOOD (the app shows "this doesn't look like food" instead of calories); otherwise the estimate is
printed. Photos are squashed to the model size with the same resize used in training.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch

from .data import load_rgb_uint8
from .export import DeployModel, load_checkpoint
from .model import TARGETS

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def expand(args: list[str]) -> list[Path]:
    """Files, folders (their images) and wildcards (PowerShell does not expand them for you)."""
    paths: list[Path] = []
    for arg in args:
        matches = [Path(m) for m in sorted(glob.glob(arg))] or [Path(arg)]
        for m in matches:
            paths += sorted(f for f in m.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES) if m.is_dir() else [m]
    return paths


def describe(values: list[float], threshold: float, has_food_output: bool) -> str:
    nutrition = ", ".join(
        f"{v:.0f} kcal" if name == "calories" else f"{name} {v:.0f} g" for name, v in zip(TARGETS, values[: len(TARGETS)])
    )
    if not has_food_output:
        return f"(model has no food output) {nutrition}"
    p = values[len(TARGETS)]
    return f"food {p:.2f} -> NOT FOOD" if p < threshold else f"food {p:.2f} -> food | {nutrition}"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("photos", nargs="+", help="image files, folders or wildcards")
    p.add_argument("--threshold", type=float, default=0.5, help="food probability below this is treated as not food")
    args = p.parse_args(argv)

    net, size = load_checkpoint(args.checkpoint)
    model = DeployModel(net).eval()
    paths = expand(args.photos)
    if not paths:
        p.error("no photos found")
    width = max(len(path.name) for path in paths)
    for path in paths:
        with torch.no_grad():
            out = model(torch.from_numpy(load_rgb_uint8(path, size)[None]))[0].tolist()
        print(f"{path.name:<{width}}  {describe(out, args.threshold, net.food_head)}")


if __name__ == "__main__":
    main()
