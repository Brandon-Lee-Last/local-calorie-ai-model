"""Lightweight multi-output network: food image -> calories, mass and macros."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

# Output order of the regression head, and the units of each (kcal / grams).
TARGETS = ("calories", "mass", "fat", "carb", "protein")
# With the food head the model has one more output: the probability that the photo shows food.
OUTPUT_NAMES = TARGETS + ("food",)

# Fixed divisors so every output regresses values of roughly unit magnitude.
# The network works in scaled units; multiply by these to get kcal / grams.
TARGET_SCALE = {
    "calories": 500.0,
    "mass": 400.0,
    "fat": 30.0,
    "carb": 40.0,
    "protein": 30.0,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# name -> (constructor, pretrained weights, feature channels)
_BACKBONES = {
    "mobilenet_v3_small": (
        models.mobilenet_v3_small,
        models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        576,
    ),
    "mobilenet_v3_large": (
        models.mobilenet_v3_large,
        models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        960,
    ),
}


class NutritionNet(nn.Module):
    """Image in, five non-negative nutrition estimates out (in scaled units), plus an optional food logit.

    Input is a float tensor ``(B, 3, H, W)`` with RGB values in ``[0, 1]``.
    ImageNet normalisation happens inside the model so the app only has to
    resize and scale pixels, which removes a common source of train/serve skew.
    """

    def __init__(
        self,
        backbone: str = "mobilenet_v3_small",
        pretrained: bool = True,
        hidden: int = 256,
        dropout: float = 0.2,
        food_head: bool = True,
    ) -> None:
        super().__init__()
        self.food_head = food_head
        if backbone not in _BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}; choose from {sorted(_BACKBONES)}")
        ctor, weights, channels = _BACKBONES[backbone]
        self.features = ctor(weights=weights if pretrained else None).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.Hardswish(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, len(TARGETS) + int(food_head))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        h = self.neck(self.pool(self.features(x)))
        out = self.head(h)
        # softplus keeps every nutrition estimate non-negative.
        values = F.softplus(out[:, : len(TARGETS)])
        if not self.food_head:
            return values
        # Last column is the food LOGIT (apply a sigmoid for the probability); the rest are scaled values.
        return torch.cat([values, out[:, len(TARGETS) :]], dim=1)


def scale_vector(device: torch.device | str = "cpu") -> torch.Tensor:
    """Per-output multipliers (shape ``(len(TARGETS),)``) from scaled to real units."""
    return torch.tensor([TARGET_SCALE[t] for t in TARGETS], device=device)


def to_real_units(scaled: torch.Tensor) -> dict[str, float]:
    """Convert one sample's scaled output ``(len(TARGETS),)`` to ``{name: value}``."""
    real = scaled.detach().cpu() * scale_vector()
    return {name: float(v) for name, v in zip(TARGETS, real)}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
