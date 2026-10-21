"""Export a trained checkpoint to ONNX (optionally int8) for on-device use.

    python -m calorie_model.export runs/default/best.pt --out build/nutrition.onnx
    python -m calorie_model.export runs/default/best.pt --out build/nutrition.onnx --quantize --calib-dir data/nutrition5k/imagery

App-facing contract (shared with the TFLite export):

    input   "image"      uint8   (N, S, S, 3)  RGB, NHWC, raw 0-255 pixels
    output  "nutrition"  float32 (N, 5)        kcal, g mass, g fat, g carb, g protein
                                               (order: calorie_model.TARGETS)
                         or (N, 6) when the model has the food head: the same five, then the
                         probability (0-1) that the photo shows food (calorie_model.OUTPUT_NAMES)

Scaling to [0, 1], ImageNet normalisation and the NHWC->NCHW transpose all
happen inside the graph, so the app only resizes to S x S and hands over bytes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import load_rgb_uint8
from .model import OUTPUT_NAMES, TARGETS, NutritionNet, scale_vector


class DeployModel(nn.Module):
    """Wraps the trained net with the app-facing input/output contract.

    With ``tta`` (test-time augmentation) the photo is also run mirrored, flipped and both, and the four
    results are averaged inside the model. That makes the answer steadier for the same photo (the model
    was trained with flips, so the four views should agree), at four times the compute; the input and
    output formats are unchanged.
    """

    def __init__(self, net: NutritionNet, tta: bool = False) -> None:
        super().__init__()
        self.net = net
        self.tta = tta
        self.register_buffer("scale", scale_vector(), persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Cast before permuting: litert-torch cannot lower permute on uint8 tensors.
        x = image.float().permute(0, 3, 1, 2) / 255.0
        if self.tta:
            x = torch.cat([x, x.flip(3), x.flip(2), x.flip(2).flip(3)], dim=0)
        out = self.net(x)
        if self.tta:
            out = out.reshape(4, -1, out.shape[1]).mean(dim=0)  # average over the four versions
        values = out[:, : len(TARGETS)] * self.scale
        if not self.net.food_head:
            return values
        return torch.cat([values, torch.sigmoid(out[:, len(TARGETS) :])], dim=1)


def load_checkpoint(path: str | Path) -> tuple[NutritionNet, int]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    net = NutritionNet(ckpt["backbone"], pretrained=False, food_head=ckpt.get("food_head", False))
    net.load_state_dict(ckpt["state_dict"])
    return net.eval(), ckpt["img_size"]


def deploy_model(model: nn.Module, tta: bool = False) -> nn.Module:
    return (DeployModel(model, tta) if isinstance(model, NutritionNet) else model).eval()


def random_pixels(n: int, img_size: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, img_size, img_size, 3), dtype=np.uint8)


def export_onnx(model: nn.Module, img_size: int, out: Path, opset: int = 17, tta: bool = False) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.from_numpy(random_pixels(1, img_size))
    torch.onnx.export(
        deploy_model(model, tta),
        dummy,
        str(out),
        input_names=["image"],
        output_names=["nutrition"],
        dynamic_axes={"image": {0: "batch"}, "nutrition": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )


def check_parity(model: nn.Module, onnx_path: Path, img_size: int, atol: float = 1e-2, tta: bool = False) -> float:
    """Max abs difference (real units) between PyTorch and ONNX Runtime outputs."""
    import onnxruntime as ort

    pixels = random_pixels(2, img_size, seed=1)
    with torch.no_grad():
        expected = deploy_model(model, tta)(torch.from_numpy(pixels)).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": pixels})[0]
    diff = float(np.abs(expected - got).max())
    if diff > atol:
        raise AssertionError(f"ONNX output differs from PyTorch by {diff:.4f} (> {atol})")
    return diff


def remove_identities(src: Path, dst: Path) -> None:
    """Rewrite an ONNX file with every non-output Identity node bypassed."""
    import onnx

    model = onnx.load(str(src))
    outputs = {o.name for o in model.graph.output}
    bypass = {n.output[0]: n.input[0] for n in model.graph.node if n.op_type == "Identity" and n.output[0] not in outputs}

    def resolve(name: str) -> str:
        while name in bypass:
            name = bypass[name]
        return name

    kept = [n for n in model.graph.node if not (n.op_type == "Identity" and n.output[0] in bypass)]
    for node in kept:
        for i, name in enumerate(node.input):
            node.input[i] = resolve(name)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    onnx.save(model, str(dst))


def quantize_int8(src: Path, dst: Path, img_size: int, calib_dir: Path | None, n_calib: int = 100) -> None:
    """Static int8 (QDQ) quantisation. Re-measure accuracy afterwards; it is not free."""
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    # The exporter routes most conv biases through Identity nodes; the quantiser only
    # quantises biases that are initializers, so strip them first.
    prepped = dst.with_name(dst.stem + ".prep.onnx")
    remove_identities(src, prepped)
    src = prepped

    if calib_dir is not None:
        paths = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg") for p in calib_dir.rglob(ext))[:n_calib]
        if not paths:
            raise FileNotFoundError(f"no images found under {calib_dir}")
        batches = [load_rgb_uint8(p, img_size)[None] for p in paths]
    else:
        print("WARNING: calibrating on random noise; pass --calib-dir with real food photos.")
        batches = list(random_pixels(16, img_size)[:, None])

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
    prepped.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--out", type=Path, default=Path("build/nutrition.onnx"))
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--calib-dir", type=Path, default=None)
    p.add_argument("--tta", action="store_true", help="average the model over the photo mirrored/flipped (steadier, 4x compute)")
    args = p.parse_args(argv)

    net, img_size = load_checkpoint(args.checkpoint)
    export_onnx(net, img_size, args.out, tta=args.tta)
    diff = check_parity(net, args.out, img_size, tta=args.tta)
    print(f"{args.out}: {args.out.stat().st_size / 1e6:.2f} MB, max |torch - onnx| = {diff:.2e}")
    print("outputs:", ", ".join(OUTPUT_NAMES if net.food_head else TARGETS))

    if args.quantize:
        q = args.out.with_name(args.out.stem + ".int8.onnx")
        quantize_int8(args.out, q, img_size, args.calib_dir)
        print(f"{q}: {q.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
