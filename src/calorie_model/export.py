"""Export a trained checkpoint to ONNX (optionally int8) for on-device use.

    python -m calorie_model.export runs/default/best.pt --out build/nutrition.onnx
    python -m calorie_model.export runs/default/best.pt --out build/nutrition.onnx --quantize --calib-dir data/nutrition5k/imagery

The exported graph takes ``image`` float32 ``(N, 3, S, S)`` with RGB in [0, 1]
(normalisation is inside the graph) and returns ``nutrition`` ``(N, 5)`` in real
units, ordered as ``calorie_model.TARGETS`` (kcal, g, g, g, g).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import load_image
from .model import TARGETS, NutritionNet, scale_vector


class DeployModel(nn.Module):
    """Wraps the trained net so the exported graph emits kcal/grams directly."""

    def __init__(self, net: NutritionNet) -> None:
        super().__init__()
        self.net = net
        self.register_buffer("scale", scale_vector(), persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image) * self.scale


def load_checkpoint(path: str | Path) -> tuple[NutritionNet, int]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    net = NutritionNet(ckpt["backbone"], pretrained=False)
    net.load_state_dict(ckpt["state_dict"])
    return net.eval(), ckpt["img_size"]


def export_onnx(model: nn.Module, img_size: int, out: Path, opset: int = 17) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    model = DeployModel(model).eval() if isinstance(model, NutritionNet) else model.eval()
    dummy = torch.rand(1, 3, img_size, img_size)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["image"],
        output_names=["nutrition"],
        dynamic_axes={"image": {0: "batch"}, "nutrition": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )


def check_parity(model: nn.Module, onnx_path: Path, img_size: int, atol: float = 1e-2) -> float:
    """Max abs difference (real units) between PyTorch and ONNX Runtime outputs."""
    import onnxruntime as ort

    model = DeployModel(model).eval() if isinstance(model, NutritionNet) else model.eval()
    x = torch.rand(2, 3, img_size, img_size)
    with torch.no_grad():
        expected = model(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": x.numpy()})[0]
    diff = float(np.abs(expected - got).max())
    if diff > atol:
        raise AssertionError(f"ONNX output differs from PyTorch by {diff:.4f} (> {atol})")
    return diff


def quantize_int8(src: Path, dst: Path, img_size: int, calib_dir: Path | None, n_calib: int = 100) -> None:
    """Static int8 (QDQ) quantisation. Re-measure accuracy afterwards; it is not free."""
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    if calib_dir is not None:
        paths = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg") for p in calib_dir.rglob(ext))[:n_calib]
        if not paths:
            raise FileNotFoundError(f"no images found under {calib_dir}")
        batches = [load_image(p, img_size).unsqueeze(0).numpy() for p in paths]
    else:
        print("WARNING: calibrating on random noise; pass --calib-dir with real food photos.")
        batches = [np.random.rand(1, 3, img_size, img_size).astype(np.float32) for _ in range(16)]

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._it = iter({"image": b} for b in batches)

        def get_next(self):
            return next(self._it, None)

    quantize_static(
        str(src),
        str(dst),
        Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--out", type=Path, default=Path("build/nutrition.onnx"))
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--calib-dir", type=Path, default=None)
    args = p.parse_args(argv)

    net, img_size = load_checkpoint(args.checkpoint)
    export_onnx(net, img_size, args.out)
    diff = check_parity(net, args.out, img_size)
    print(f"{args.out}: {args.out.stat().st_size / 1e6:.2f} MB, max |torch - onnx| = {diff:.2e}")
    print("outputs:", ", ".join(TARGETS))

    if args.quantize:
        q = args.out.with_name(args.out.stem + ".int8.onnx")
        quantize_int8(args.out, q, img_size, args.calib_dir)
        print(f"{q}: {q.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
