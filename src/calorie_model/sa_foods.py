"""South African dish list and per-100 g nutrient sheet, plus the arithmetic that turns a weighed plate into labels.

    datasets/south_african/dishes.csv             dish names, aliases and search terms (no numbers)
    datasets/south_african/nutrition_per_100g.csv per-100 g values, blank until someone fills them in from a source

The sheet ships empty on purpose: no value in it may come from memory. Every filled row needs a ``source``
(a SAFOODS table, a product label, a lab or dietitian result) so a number can be traced and challenged.
"""
from __future__ import annotations

import csv
import difflib
import math
from dataclasses import dataclass
from pathlib import Path

from .model import TARGETS

NUTRIENTS = tuple(t for t in TARGETS if t != "mass")  # calories, fat, carb, protein

DATA_DIR = Path(__file__).resolve().parents[2] / "datasets" / "south_african"
DEFAULT_TAXONOMY = DATA_DIR / "dishes.csv"
DEFAULT_TABLE = DATA_DIR / "nutrition_per_100g.csv"

# Atwater factors: kcal per gram of protein, carbohydrate and fat. A row whose stated energy is far from what
# its own macros imply is probably a typo or a unit mix-up.
KCAL_PER_G = {"protein": 4.0, "carb": 4.0, "fat": 9.0}
ENERGY_TOLERANCE = 0.25  # fraction (fibre, alcohol and rounding make exact agreement unusual)
MAX_COMPONENT_GRAMS = 5000.0


@dataclass(frozen=True)
class Dish:
    dish_id: str
    name: str
    aliases: tuple[str, ...]
    category: str
    description: str
    queries: tuple[str, ...]


def _split(text: str | None, sep: str) -> tuple[str, ...]:
    return tuple(p.strip() for p in (text or "").split(sep) if p.strip())


def read_taxonomy(path: str | Path = DEFAULT_TAXONOMY) -> dict[str, Dish]:
    dishes: dict[str, Dish] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            dish_id = (r.get("dish_id") or "").strip()
            if not dish_id:
                raise ValueError(f"{path}: a row has no dish_id")
            if dish_id in dishes:
                raise ValueError(f"{path}: duplicate dish_id {dish_id!r}")
            dishes[dish_id] = Dish(
                dish_id,
                (r.get("name") or dish_id).strip(),
                _split(r.get("aliases"), "|"),
                (r.get("category") or "").strip(),
                (r.get("description") or "").strip(),
                _split(r.get("commons_queries"), ";"),
            )
    return dishes


def resolve_dish(text: str, taxonomy: dict[str, Dish]) -> str:
    """Map a dish id, name or alias to its dish id; raise with a suggestion when nothing matches."""
    key = text.strip().lower()
    if key in taxonomy:
        return key
    for dish in taxonomy.values():
        if key == dish.name.lower() or key in (a.lower() for a in dish.aliases):
            return dish.dish_id
    close = difflib.get_close_matches(key, list(taxonomy), n=3, cutoff=0.6)
    hint = f" (did you mean {', '.join(close)}?)" if close else ""
    raise ValueError(f"unknown dish {text!r}{hint}; add it to dishes.csv first")


@dataclass(frozen=True)
class Per100g:
    values: dict[str, float]  # calories / fat / carb / protein, NaN where blank
    source: str
    source_ref: str

    def complete(self) -> bool:
        return all(not math.isnan(v) for v in self.values.values())


def read_table(path: str | Path = DEFAULT_TABLE, taxonomy: dict[str, Dish] | None = None) -> dict[str, Per100g]:
    """Rows with at least one value. Raises on a value without a source, a negative or non-numeric value,
    a dish id missing from the taxonomy, or a duplicate row."""
    table: dict[str, Per100g] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            dish_id = (r.get("dish_id") or "").strip()
            values = {}
            for n in NUTRIENTS:
                text = (r.get(n) or "").strip()
                if not text:
                    values[n] = float("nan")
                    continue
                try:
                    value = float(text)
                except ValueError:
                    raise ValueError(f"{path}: {dish_id}: {n} {text!r} is not a number") from None
                if value < 0 or math.isnan(value):
                    raise ValueError(f"{path}: {dish_id}: {n} must be a non-negative number, got {text!r}")
                values[n] = value
            if all(math.isnan(v) for v in values.values()):
                continue  # a blank row: nothing to use yet
            if taxonomy is not None and dish_id not in taxonomy:
                raise ValueError(f"{path}: {dish_id!r} is not in the dish list")
            if dish_id in table:
                raise ValueError(f"{path}: duplicate row for {dish_id!r}")
            source = (r.get("source") or "").strip()
            if not source:
                raise ValueError(f"{path}: {dish_id} has values but no source; say where each number comes from")
            table[dish_id] = Per100g(values, source, (r.get("source_ref") or "").strip())
    return table


def energy_warnings(table: dict[str, Per100g]) -> list[str]:
    """Rows whose calories disagree with their own macros by more than ``ENERGY_TOLERANCE``."""
    out = []
    for dish_id, row in table.items():
        if not row.complete():
            continue
        v = row.values
        implied = sum(KCAL_PER_G[k] * v[k] for k in KCAL_PER_G)
        stated = v["calories"]
        if abs(stated - implied) > ENERGY_TOLERANCE * max(stated, implied, 1.0):
            out.append(f"{dish_id}: says {stated:.0f} kcal per 100 g but fat/carb/protein add up to about {implied:.0f}")
    return out


def parse_components(text: str, taxonomy: dict[str, Dish]) -> list[tuple[str, float]]:
    """``"pap:250|boerewors:150"`` -> ``[("pap", 250.0), ("boerewors", 150.0)]`` (grams as served)."""
    parts = [p for p in (s.strip() for s in text.split("|")) if p]
    if not parts:
        raise ValueError("no components")
    out: list[tuple[str, float]] = []
    for part in parts:
        name, sep, grams = part.rpartition(":")
        if not sep or not name.strip():
            raise ValueError(f"component {part!r} should look like dish_id:grams")
        try:
            g = float(grams)
        except ValueError:
            raise ValueError(f"component {part!r}: {grams.strip()!r} is not a number of grams") from None
        if not 0 < g <= MAX_COMPONENT_GRAMS:
            raise ValueError(f"component {part!r}: grams must be between 0 and {MAX_COMPONENT_GRAMS:.0f}")
        out.append((resolve_dish(name, taxonomy), g))
    return out


def meal_totals(components: list[tuple[str, float]], table: dict[str, Per100g]) -> dict[str, float]:
    """Mass (always known) and, per nutrient, the sum over components of grams x per-100 g / 100. A nutrient is
    NaN when any component has no value for it, because a partial sum would under-count the plate."""
    totals = {"mass": sum(g for _, g in components)}
    for n in NUTRIENTS:
        parts = [table[d].values[n] * g / 100.0 if d in table else float("nan") for d, g in components]
        totals[n] = float("nan") if any(math.isnan(p) for p in parts) else sum(parts)
    return totals
