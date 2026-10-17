"""In-memory image batches with GPU-side augmentation.

Decoding JPEGs and augmenting them in DataLoader worker processes managed only
~80 images/s on Windows (the per-image work is ~8 ms; the multi-process loader was
the bottleneck, and it also crashed with ``WinError 6`` at the end of a run).
Instead every image is decoded once into a uint8 tensor that lives on the training
device, and batches are augmented there. The dataset is a few GB at most.

Augmentation mirrors what the CPU pipeline did: a mild random crop and zoom
(apparent size is a portion cue, so it stays gentle), random flips, a random
rotation of up to 30 degrees half the time, and brightness / contrast /
saturation / hue jitter.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# RGB -> YIQ; hue jitter is a rotation of the (I, Q) plane.
_YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])


def load_bank(ds: Dataset, device: torch.device | str, threads: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode every image of ``ds`` once: uint8 ``(N, 3, S, S)`` images and ``(N, 5)`` targets on ``device``.

    ``ds`` must return float images in [0, 1] from uint8 sources (so the round trip
    to uint8 is exact) and provide ``targets()``.
    """
    first, _ = ds[0]
    images = torch.empty((len(ds), *first.shape), dtype=torch.uint8)

    def fill(i: int) -> None:
        images[i] = (ds[i][0] * 255).round().to(torch.uint8)

    with ThreadPoolExecutor(max_workers=max(threads, 1)) as pool:
        list(pool.map(fill, range(len(ds))))
    return images.to(device), ds.targets().to(device)


def identity_params(b: int, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    one, zero = torch.ones(b, device=device), torch.zeros(b, device=device)
    return {
        "w": one, "h": one, "cx": zero, "cy": zero, "flip_x": one, "flip_y": one, "angle": zero,
        "brightness": one, "contrast": one, "saturation": one, "hue": zero,
    }  # fmt: skip


def sample_params(b: int, device: torch.device | str) -> dict[str, torch.Tensor]:
    def uniform(lo: float, hi: float) -> torch.Tensor:
        return torch.empty(b, device=device).uniform_(lo, hi)

    def coin(p: float) -> torch.Tensor:
        return (torch.rand(b, device=device) < p).float()

    area, ratio = uniform(0.85, 1.0), uniform(0.9, 1.1)
    w, h = (area * ratio).sqrt().clamp(max=1.0), (area / ratio).sqrt().clamp(max=1.0)
    return {
        "w": w,
        "h": h,
        "cx": uniform(-1, 1) * (1 - w),
        "cy": uniform(-1, 1) * (1 - h),
        "flip_x": 1 - 2 * coin(0.5),
        "flip_y": 1 - 2 * coin(0.5),
        "angle": torch.deg2rad(uniform(-30, 30)) * coin(0.5),
        "brightness": uniform(0.8, 1.2),
        "contrast": uniform(0.8, 1.2),
        "saturation": uniform(0.8, 1.2),
        "hue": uniform(-0.02, 0.02),
    }


def _gray(x: torch.Tensor) -> torch.Tensor:
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)


def augment(x: torch.Tensor, p: dict[str, torch.Tensor]) -> torch.Tensor:
    """Apply per-sample parameters ``p`` to a float ``(B, 3, S, S)`` batch in [0, 1]."""
    b = x.shape[0]
    cos, sin = p["angle"].cos(), p["angle"].sin()
    sx, sy = p["w"] * p["flip_x"], p["h"] * p["flip_y"]
    # Maps output coordinates to the input region: a crop of fractional size (w, h)
    # centred on (cx, cy), mirrored and rotated. Out-of-image samples are black.
    theta = torch.stack(
        [torch.stack([cos * sx, -sin * sy, p["cx"]], 1), torch.stack([sin * sx, cos * sy, p["cy"]], 1)], 1
    )
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    v = lambda name: p[name].view(b, 1, 1, 1)  # noqa: E731
    x = (x * v("brightness")).clamp(0, 1)
    mean = _gray(x).mean((1, 2, 3), keepdim=True)
    x = ((x - mean) * v("contrast") + mean).clamp(0, 1)
    gray = _gray(x)
    x = ((x - gray) * v("saturation") + gray).clamp(0, 1)

    phi = p["hue"] * 2 * math.pi
    rot = torch.zeros(b, 3, 3, device=x.device)
    rot[:, 0, 0] = 1
    rot[:, 1, 1], rot[:, 1, 2] = phi.cos(), -phi.sin()
    rot[:, 2, 1], rot[:, 2, 2] = phi.sin(), phi.cos()
    yiq = _YIQ.to(x.device)
    mix = torch.linalg.inv(yiq) @ rot @ yiq
    return torch.einsum("bij,bjhw->bihw", mix, x).clamp(0, 1)


class Batches:
    """Iterate ``(images, targets)`` batches from an in-memory bank; optionally shuffled and augmented."""

    def __init__(
        self, images: torch.Tensor, targets: torch.Tensor, batch_size: int, shuffle: bool = False, augment_: bool = False
    ) -> None:
        self.images, self.targets = images, targets
        self.batch_size, self.shuffle, self.augment = batch_size, shuffle, augment_
        self.drop_last = shuffle and len(images) > batch_size

    def __len__(self) -> int:
        n = len(self.images)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def __iter__(self):
        device = self.images.device
        n = len(self.images)
        order = torch.randperm(n, device=device) if self.shuffle else torch.arange(n, device=device)
        for i in range(len(self)):
            idx = order[i * self.batch_size : (i + 1) * self.batch_size]
            x = self.images[idx].float().div_(255.0)
            if self.augment:
                x = augment(x, sample_params(len(idx), device))
            yield x, self.targets[idx]
