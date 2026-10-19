"""Datasets and image preprocessing."""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
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
        resized = T.Resize((img_size, img_size))(ImageOps.exif_transpose(im).convert("RGB"))  # phones store portrait shots rotated + a tag
    return np.array(resized, dtype=np.uint8)  # a copy: PIL's buffer view is read-only


def _scale() -> torch.Tensor:
    return torch.tensor([TARGET_SCALE[t] for t in TARGETS])


# Training targets are TARGET_WIDTH wide:
#   0-4    the five scaled nutrition values (0 where unknown)
#   5      food: 1 for a food photo, 0 for anything else
#   6-10   which of the five values are real labels (1) or unknown (0), so a source that only has weights,
#          or only calories, still trains those outputs and nothing else
#   11     weight of this photo in the nutrition loss: 1 for trusted labels, less for noisy sources
N_OUT = len(TARGETS)
FOOD_COL, KNOWN_START = N_OUT, N_OUT + 1
WEIGHT_COL = KNOWN_START + N_OUT
TARGET_WIDTH = WEIGHT_COL + 1


def make_target(nutrition: torch.Tensor | None = None, food: float = 1.0, weight: float = 1.0) -> torch.Tensor:
    """Build one training target. ``nutrition`` holds the five scaled values with NaN for unknown ones,
    or is None when the photo has no nutrition labels at all (food / not-food photos)."""
    if nutrition is None:
        values, known = torch.zeros(N_OUT), torch.zeros(N_OUT)
    else:
        known = (~torch.isnan(nutrition)).float()
        values = torch.nan_to_num(nutrition, nan=0.0)
    return torch.cat([values, torch.tensor([food]), known, torch.tensor([weight])])


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

        chosen = [r for r in rows if r["dish_id"] in keep and r["view"] in views]
        self.samples = [(root / r["path"], make_target(labels[r["dish_id"]])) for r in chosen]
        self.dishes = [r["dish_id"] for r in chosen]  # parallel to samples: which dish each photo shows
        self.views = [r["view"] for r in chosen]
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


class OpenImagesDataset(Dataset):
    """Food / not-food photos from the cache built by ``calorie_model.prepare_openimages``.

    There are no nutrition labels, so these images train only the food output. ``split`` is
    ``train``, ``val`` (a fixed share of the training pool) or ``test`` (the held-out Open Images
    validation subset). ``groups`` optionally keeps only some of food / animal / person / other.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        img_size: int = 224,
        val_fraction: float = 0.1,
        seed: int = 0,
        groups: tuple[str, ...] | None = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val' or 'test'")
        root = Path(root)
        self.transform = eval_transform(img_size)
        with open(root / "index.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        pool = "test" if split == "test" else "train"
        ids = sorted(r["image_id"] for r in rows if r["split"] == pool)
        keep = set(ids)
        if split != "test":
            n_val = max(1, round(len(ids) * val_fraction)) if val_fraction > 0 and len(ids) > 1 else 0
            val = set(random.Random(seed).sample(ids, n_val))
            keep = val if split == "val" else keep - val

        chosen = [r for r in rows if r["image_id"] in keep and (groups is None or r["group"] in groups)]
        self.samples = [
            (root / r["path"], make_target(None, food=1.0 if r["label"] == "food" else 0.0))
            for r in chosen
        ]
        self.groups = [r["group"] for r in chosen]
        if not self.samples:
            raise RuntimeError(f"no images for split {split!r} under {root}; run prepare_openimages first")

    def targets(self) -> torch.Tensor:
        return torch.stack([t for _, t in self.samples])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, target = self.samples[i]
        with Image.open(path) as im:
            return self.transform(im.convert("RGB")), target


class ExtraNutritionDataset(Dataset):
    """Any extra food photos with nutrition labels, which may be partial or noisy.

    ``root/index.csv`` has one row per photo. Columns (only ``path`` is required):

    * ``path``: image file, relative to ``root`` (any size; it is decoded and squashed to ``img_size``)
    * ``calories, mass, fat, carb, protein``: real units (kcal and grams); leave blank when unknown
    * ``weight``: how much to trust the photo's labels, default 1 (use less for AI-generated or
      estimated labels so they steer the model without dominating)
    * ``split``: ``test`` keeps the photo out of training and reports it separately; anything else
      (default) is training data
    * ``dish_id`` and ``source``: informational (default: the path and the folder name)

    Every photo here counts as food. See ``docs/adding-data.md``.
    """

    def __init__(self, root: str | Path, split: str, img_size: int = 224) -> None:
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        root = Path(root)
        self.size = img_size
        self.transform = eval_transform(img_size)
        scale = _scale()
        with open(root / "index.csv", newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if ((r.get("split") or "train").strip().lower() == "test") == (split == "test")]

        def number(row: dict[str, str], key: str) -> float:
            text = (row.get(key) or "").strip()
            if not text:
                return float("nan")
            value = float(text)
            if value < 0:
                raise ValueError(f"{root / 'index.csv'}: negative {key} {value} for {row['path']}")
            return value

        self.samples, self.sources = [], []
        for r in rows:
            weight = float((r.get("weight") or "1").strip() or 1)
            if weight <= 0:
                raise ValueError(f"{root / 'index.csv'}: weight must be > 0 for {r['path']}")
            nutrition = torch.tensor([number(r, t) for t in TARGETS]) / scale
            self.samples.append((root / r["path"], make_target(nutrition, food=1.0, weight=weight)))
            self.sources.append((r.get("source") or root.name).strip())
        if not self.samples:
            raise RuntimeError(f"no {split!r} photos in {root / 'index.csv'}")

    def targets(self) -> torch.Tensor:
        return torch.stack([t for _, t in self.samples])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, target = self.samples[i]
        with Image.open(path) as im:
            if im.format == "JPEG":
                im.draft("RGB", (self.size * 2, self.size * 2))  # decode big photos at reduced size, much faster
            return self.transform(ImageOps.exif_transpose(im).convert("RGB")), target


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
        return make_target(torch.tensor([r, (r + g) / 2, g, b, (g + b) / 2]))

    def __len__(self) -> int:
        return len(self.params)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        r, g, b = self.params[i]
        img = torch.stack([torch.full((self.img_size, self.img_size), c) for c in (r, g, b)])
        return img, self._target(r, g, b)
