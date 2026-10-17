"""Run an exported ONNX model on one image and print nutrition estimates as JSON.

    python -m calorie_model.predict build/nutrition.onnx photo.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import load_image
from .model import TARGETS

UNITS = {"calories": "kcal", "mass": "g", "fat": "g", "carb": "g", "protein": "g"}


def predict(model_path: str | Path, image_path: str | Path) -> dict[str, float]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    _, _, h, w = sess.get_inputs()[0].shape
    if not isinstance(h, int) or h != w:
        raise ValueError(f"expected a square static input size, got {(h, w)}")
    x = load_image(image_path, h).unsqueeze(0).numpy().astype(np.float32)
    out = sess.run(None, {"image": x})[0][0]
    return {name: round(float(v), 1) for name, v in zip(TARGETS, out)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", type=Path)
    p.add_argument("image", type=Path)
    args = p.parse_args(argv)
    result = predict(args.model, args.image)
    print(json.dumps({k: {"value": v, "unit": UNITS[k]} for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
