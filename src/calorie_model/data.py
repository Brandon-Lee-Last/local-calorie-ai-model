"""Datasets and image preprocessing."""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .model import TARGET_SCALE, TARGETS


def eval_transform(img_size: int) -> T.Compose:
    """Deterministic preprocessing; the app must reproduce exactly this."""
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])


def load_rgb_uint8(path: str | Path, img_size: int) -> np.ndarray:
    """Load one image as a ``(img_size, img_size, 3)`` uint8 RGB array.

    This is the exact input contract of the exported model (minus the batch
    axis) and uses the same resize as ``eval_transform``.
    """
    with Image.open(path) as im:
        resized = T.Resize((img_size, img_size))(im.convert("RGB"))
    return np.asarray(resized, dtype=np.uint8)


def _scale() -> torch.Tensor:
    return torch.tensor([TARGET_SCALE[t] for t in TARGETS])


def read_labels(path: Path) -> dict[str, torch.Tensor]:
    """``labels.csv`` (real units) -> ``{dish_id: scaled target}`` in ``TARGETS`` order."""
    with open(path, newline="") as f:
        return {r["dish_id"]: torch.tensor([float(r[t]) for t in TARGETS]) / _scale() for r in csv.DictReader(f)}


class Nutrition5kDataset(Dataset):
    """Nutrition5k images from the cache built by ``calorie_model.prepare_data``.

    ``split`` is ``train``, ``val`` or ``test``. ``test`` is the official test split.
    ``val`` is a fixed ``val_fraction`` of the *dishes* in the training pool, held
    out so that different views of one dish never straddle train and val.
    ``views`` selects overhead photos, side-angle frames or both. ``train_fraction``
    keeps only that share of the training dishes (never of val or test).
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        img_size: int = 224,
        views: tuple[str, ...] = ("overhead", "side"),
        val_fraction: float = 0.1,
        seed: int = 0,
        train_fraction: float = 1.0,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val' or 'test'")
        root = Path(root)
        self.transform = eval_transform(img_size)

        labels = read_labels(root / "labels.csv")
        with open(root / "index.csv", newline="") as f:
            rows = list(csv.DictReader(f))

        pool = "test" if split == "test" else "train"
        dishes = sorted({r["dish_id"] for r in rows if r["split"] == pool and r["dish_id"] in labels})
        keep = set(dishes)
        if split != "test":
            n_val = max(1, round(len(dishes) * val_fraction)) if val_fraction > 0 and len(dishes) > 1 else 0
            val = set(random.Random(seed).sample(dishes, n_val))
            keep = val if split == "val" else keep - val
            if split == "train" and train_fraction < 1.0:  # fewer distinct dishes, for learning curves
                n_keep = max(1, round(len(keep) * train_fraction))
                keep = set(random.Random(seed + 1).sample(sorted(keep), n_keep))

        self.samples = [
            (root / r["path"], labels[r["dish_id"]]) for r in rows if r["dish_id"] in keep and r["view"] in views
        ]
        if not self.samples:
            raise RuntimeError(f"no images for split {split!r}, views {views} under {root}; run prepare_data first")

    def targets(self) -> torch.Tensor:
        return torch.stack([t for _, t in self.samples])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, target = self.samples[i]
        with Image.open(path) as im:
            return self.transform(im.convert("RGB")), target


class SyntheticDataset(Dataset):
    """Procedural images whose labels are a function of pixel statistics.

    Exists so the whole train/export/predict pipeline can be smoke-tested
    without downloading anything. It says nothing about real-world accuracy.
    """

    def __init__(self, n: int = 64, img_size: int = 224, seed: int = 0) -> None:
        rng = random.Random(seed)
        self.img_size = img_size
        self.params = [(rng.random(), rng.random(), rng.random()) for _ in range(n)]

    def targets(self) -> torch.Tensor:
        return torch.stack([self._target(*p) for p in self.params])

    @staticmethod
    def _target(r: float, g: float, b: float) -> torch.Tensor:
        # In scaled units: each output tracks one colour channel or a mix.
        return torch.tensor([r, (r + g) / 2, g, b, (g + b) / 2])

    def __len__(self) -> int:
        return len(self.params)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        r, g, b = self.params[i]
        img = torch.stack([torch.full((self.img_size, self.img_size), c) for c in (r, g, b)])
        return img, self._target(r, g, b)
