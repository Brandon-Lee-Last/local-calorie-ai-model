"""Time training steps on this machine's GPU to find fast settings.

    python -m calorie_model.bench

Prints the cost of the GPU augmentation and of a full training step for every
combination of cuDNN autotuning, channels-last memory layout and mixed precision.
No data is needed; it runs on random tensors and takes about a minute.
"""
from __future__ import annotations

import argparse
import time

import torch
from torch import nn

from .batches import augment, sample_params
from .model import NutritionNet


def time_it(fn, steps: int, device: torch.device) -> float:
    """Seconds per call, after a short warm-up (lazy init, cuDNN autotune)."""
    for _ in range(3):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / steps


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--steps", type=int, default=20)
    args = p.parse_args(argv)

    device = torch.device(args.device)
    b = args.batch_size
    x = torch.rand(b, 3, args.img_size, args.img_size, device=device)
    y = torch.rand(b, 5, device=device)

    t = time_it(lambda: augment(x, sample_params(b, device)), args.steps, device)
    print(f"device {device}, batch {b}, {args.img_size}px")
    print(f"augmentation alone: {t * 1000:6.1f} ms/batch ({b / t:7.0f} img/s)\n")

    print(f"{'cudnn.benchmark':>15} {'channels_last':>13} {'amp':>5} {'ms/step':>9} {'img/s':>8}")
    for benchmark in (False, True):
        for channels_last in (False, True):
            for amp in (False, True):
                if device.type != "cuda" and (benchmark or channels_last or amp):
                    continue  # these options only apply on CUDA
                torch.backends.cudnn.benchmark = benchmark
                model = NutritionNet(pretrained=False, food_head=False).to(device)  # step cost is the same either way
                if channels_last:
                    model = model.to(memory_format=torch.channels_last)
                xb = x.contiguous(memory_format=torch.channels_last) if channels_last else x
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
                scaler = torch.amp.GradScaler("cuda", enabled=amp)

                def step() -> None:
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                        pred = model(xb)
                    loss = nn.functional.smooth_l1_loss(pred.float(), y)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()

                t = time_it(step, args.steps, device)
                print(f"{str(benchmark):>15} {str(channels_last):>13} {str(amp):>5} {t * 1000:9.1f} {b / t:8.0f}")


if __name__ == "__main__":
    main()
