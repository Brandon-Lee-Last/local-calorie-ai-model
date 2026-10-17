"""Datasets and image preprocessing."""
from __future__ import annotations

import csv
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .model import TARGET_SCALE, TARGETS


def eval_transform(img_size: int) -> T.Compose:
    """Deterministic preprocessing; the app must reproduce exactly this."""
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])


def train_transform(img_size: int) -> T.Compose:
    # Overhead plates have no canonical orientation, so flips/rotations are
    # label-preserving. Scale jitter is kept mild because apparent size is a
    # portion cue and aggressive crops would corrupt the mass/calorie label.
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply([T.RandomRotation(30)], p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            T.ToTensor(),
        ]
    )


def load_image(path: str | Path, img_size: int) -> torch.Tensor:
    """Load one image as a ``(3, img_size, img_size)`` float tensor in [0, 1]."""
    with Image.open(path) as im:
        return eval_transform(img_size)(im.convert("RGB"))


class Nutrition5kDataset(Dataset):
    """Nutrition5k overhead RGB images with dish-level calorie/mass/macro labels.

    Expects the layout produced by ``gsutil -m cp -r gs://nutrition5k_dataset/...``::

        root/metadata/dish_metadata_cafe1.csv        (and cafe2)
        root/imagery/realsense_overhead/<dish_id>/rgb.png
        root/dish_ids/splits/rgb_{train,test}_ids.txt

    Verify this against your download; ``scripts/download_nutrition5k.sh`` fetches it.
    The metadata CSVs have no header and a variable number of per-ingredient
    columns after the first six, which are:
    ``dish_id, calories, mass, fat, carb, protein``.
    """

    def __init__(self, root: str | Path, split: str, img_size: int = 224, train: bool = False) -> None:
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.root = Path(root)
        self.transform = train_transform(img_size) if train else eval_transform(img_size)

        labels = self._read_labels()
        ids_file = self.root / "dish_ids" / "splits" / f"rgb_{split}_ids.txt"
        ids = [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]
        self.samples = []
        for dish_id in ids:
            img = self.root / "imagery" / "realsense_overhead" / dish_id / "rgb.png"
            # A few dishes have labels but no overhead image (or vice versa); skip them.
            if dish_id in labels and img.exists():
                self.samples.append((img, labels[dish_id]))
        if not self.samples:
            raise RuntimeError(f"no usable samples found under {self.root} for split {split!r}")

    def _read_labels(self) -> dict[str, torch.Tensor]:
        # CSV column order is calories, mass, fat, carb, protein == TARGETS order.
        labels = {}
        for csv_path in sorted((self.root / "metadata").glob("dish_metadata_cafe*.csv")):
            with open(csv_path, newline="") as f:
                for row in csv.reader(f):
                    if len(row) < 6:
                        continue
                    values = [float(v) for v in row[1:6]]
                    labels[row[0]] = torch.tensor(values) / _scale()
        return labels

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

    def __len__(self) -> int:
        return len(self.params)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        r, g, b = self.params[i]
        img = torch.stack([torch.full((self.img_size, self.img_size), c) for c in (r, g, b)])
        # In scaled units: each output tracks one colour channel or a mix.
        target = torch.tensor([r, (r + g) / 2, g, b, (g + b) / 2])
        return img, target


def _scale() -> torch.Tensor:
    return torch.tensor([TARGET_SCALE[t] for t in TARGETS])
