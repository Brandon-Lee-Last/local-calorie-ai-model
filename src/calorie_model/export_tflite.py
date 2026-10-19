"""Export a trained checkpoint to LiteRT/TFLite for react-native-fast-tflite on Android.

    python -m calorie_model.export_tflite runs/default/best.pt --out build/nutrition.tflite

Same app-facing contract as ``calorie_model.export`` (uint8 NHWC pixels in, kcal/grams
out), with the batch size fixed to 1. Needs the converter, which is not a core
dependency and prefers Python 3.11:

    pip install litert-torch ai-edge-litert
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .export import deploy_model, load_checkpoint, random_pixels
from .model import TARGETS


def export_tflite(model: nn.Module, img_size: int, out: Path, tta: bool = False) -> None:
    try:
        import litert_torch
    except ImportError as e:  # pragma: no cover - depends on optional install
        raise SystemExit("needs the converter: pip install litert-torch (Python 3.11 recommended)") from e

    out.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.from_numpy(random_pixels(1, img_size))
    with torch.no_grad():
        edge_model = litert_torch.convert(deploy_model(model, tta), (sample,))
    edge_model.export(str(out))


def run_tflite(tflite_path: Path, pixels: np.ndarray) -> np.ndarray:
    """Run one ``(1, S, S, 3)`` uint8 batch through the .tflite file on the host CPU."""
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    if inp["dtype"] != np.uint8:
        raise AssertionError(f"expected a uint8 input tensor, model has {inp['dtype']}")
    interp.set_tensor(inp["index"], pixels)
    interp.invoke()
    return interp.get_tensor(interp.get_output_details()[0]["index"])


def check_parity(model: nn.Module, tflite_path: Path, img_size: int, atol: float = 1e-1, tta: bool = False) -> float:
    """Max abs difference (real units) between PyTorch and the .tflite on host CPU."""
    worst = 0.0
    for seed in (1, 2, 3):
        pixels = random_pixels(1, img_size, seed=seed)
        with torch.no_grad():
            expected = deploy_model(model, tta)(torch.from_numpy(pixels)).numpy()
        worst = max(worst, float(np.abs(expected - run_tflite(tflite_path, pixels)).max()))
    if worst > atol:
        raise AssertionError(f"TFLite output differs from PyTorch by {worst:.4f} (> {atol})")
    return worst


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--out", type=Path, default=Path("build/nutrition.tflite"))
    p.add_argument("--tta", action="store_true", help="average the model over the photo mirrored/flipped (steadier, 4x compute)")
    args = p.parse_args(argv)

    net, img_size = load_checkpoint(args.checkpoint)
    export_tflite(net, img_size, args.out, tta=args.tta)
    diff = check_parity(net, args.out, img_size, tta=args.tta)
    print(f"{args.out}: {args.out.stat().st_size / 1e6:.2f} MB, max |torch - tflite| = {diff:.2e}")
    print("outputs:", ", ".join(TARGETS))


if __name__ == "__main__":
    main()
