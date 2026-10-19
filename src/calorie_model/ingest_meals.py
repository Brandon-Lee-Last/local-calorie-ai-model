"""Turn weighed phone photos of meals into a training folder (the ``--extra-data`` format).

    python -m calorie_model.ingest_meals --meals meals.csv --photos-dir phone_photos --out data/sa_meals

``meals.csv`` has one row per *meal* (a plate you weighed), however many photos you took of it:

    meal_id,photos,components,calories,fat,carb,protein
    m001,IMG_0001.jpg|IMG_0002.jpg,pap:250|boerewors:150|chakalaka:80,,,,
    m002,IMG_0003.jpg,bunny_chow:640,1480,,,

m002's calories (1480) were read off a label, so they are used as given.

* ``photos``: files under ``--photos-dir``, separated by ``|``
* ``components``: ``dish_id:grams`` as served, separated by ``|``. Dish ids, names and aliases come from
  ``datasets/south_african/dishes.csv``. Weigh each part separately (tare the plate), not the whole plate once.
* ``calories, fat, carb, protein`` are optional measured totals for the whole meal (a product label, a lab).
  Where they are blank, the totals are worked out from the per-100 g sheet
  (``datasets/south_african/nutrition_per_100g.csv``) when every component has a sourced value, and left
  blank otherwise. The weighed mass is always used, so a meal trains the mass output even with no sheet at all.

Every photo is re-encoded without metadata (no GPS position or camera serial leaves the phone folder), turned
upright using its rotation tag, and shrunk. Meals, not photos, are split into train and test so that two photos of
one plate never straddle both.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

from PIL import Image, ImageOps

from .sa_foods import (
    DEFAULT_TABLE,
    DEFAULT_TAXONOMY,
    NUTRIENTS,
    energy_warnings,
    meal_totals,
    parse_components,
    read_table,
    read_taxonomy,
)

SOURCE = "sa-weighed"
COLUMNS = ["path", "calories", "mass", "fat", "carb", "protein", "weight", "split", "dish_id", "source", "components"]


def is_test(key: str, fraction: float) -> bool:
    """Deterministic: the same key always lands on the same side, whatever else is in the file."""
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0x100000000
    return bucket < fraction


def save_clean_jpeg(src: Path, dst: Path, max_side: int) -> None:
    """Upright, shrunk, metadata-free copy."""
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_side, max_side))
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "JPEG", quality=90)  # Pillow writes no EXIF unless asked


def fmt(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.1f}"


def optional(row: dict[str, str], key: str, meal: str) -> float:
    text = (row.get(key) or "").strip()
    if not text:
        return float("nan")
    value = float(text)
    if value < 0 or math.isnan(value):
        raise ValueError(f"meal {meal}: {key} must be a non-negative number")
    return value


def ingest(
    meals_csv: Path,
    photos_dir: Path,
    out: Path,
    taxonomy_path: Path = DEFAULT_TAXONOMY,
    table_path: Path = DEFAULT_TABLE,
    test_fraction: float = 0.3,
    table_weight: float = 0.5,
    max_side: int = 1024,
) -> dict[str, int]:
    taxonomy = read_taxonomy(taxonomy_path)
    table = read_table(table_path, taxonomy)
    with open(meals_csv, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("meal_id") or "").strip()]
    if not rows:
        raise ValueError(f"{meals_csv}: no meals")

    out_rows: list[dict[str, str]] = []
    seen: set[str] = set()
    stats = {"meals": 0, "photos": 0, "test_meals": 0, "with_nutrition": 0}
    for r in rows:
        meal = r["meal_id"].strip()
        if meal in seen:
            raise ValueError(f"{meals_csv}: meal_id {meal!r} appears twice")
        seen.add(meal)
        try:
            components = parse_components(r.get("components") or "", taxonomy)
            photos = [p.strip() for p in (r.get("photos") or "").split("|") if p.strip()]
            if not photos:
                raise ValueError("no photos")
            for p in photos:
                if not (photos_dir / p).is_file():
                    raise ValueError(f"photo {p!r} not found under {photos_dir}")
            totals = meal_totals(components, table)
            measured = {n: optional(r, n, meal) for n in NUTRIENTS}
        except ValueError as e:
            raise ValueError(f"{meals_csv}: meal {meal}: {e}") from None

        derived = False
        for n in NUTRIENTS:
            if not math.isnan(measured[n]):
                totals[n] = measured[n]
            elif not math.isnan(totals[n]):
                derived = True
        weight = table_weight if derived else 1.0
        split = "test" if is_test(meal, test_fraction) else ""
        stats["meals"] += 1
        stats["test_meals"] += bool(split)
        stats["with_nutrition"] += any(not math.isnan(totals[n]) for n in NUTRIENTS)
        for k, p in enumerate(photos, 1):
            dst = out / "img" / f"{meal}_{k}.jpg"
            save_clean_jpeg(photos_dir / p, dst, max_side)
            out_rows.append({
                "path": f"img/{dst.name}",
                "calories": fmt(totals["calories"]),
                "mass": fmt(totals["mass"]),
                "fat": fmt(totals["fat"]),
                "carb": fmt(totals["carb"]),
                "protein": fmt(totals["protein"]),
                "weight": f"{weight:g}",
                "split": split,
                "dish_id": meal,
                "source": SOURCE,
                "components": "|".join(f"{d}:{g:g}" for d, g in components),
            })
            stats["photos"] += 1

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(out_rows)
    return stats


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meals", type=Path, required=True, help="meals.csv, see above")
    p.add_argument("--photos-dir", type=Path, required=True, help="folder holding the photos named in meals.csv")
    p.add_argument("--out", type=Path, required=True, help="output folder to pass to train --extra-data")
    p.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    p.add_argument("--table", type=Path, default=DEFAULT_TABLE, help="per-100 g nutrient sheet")
    p.add_argument("--test-fraction", type=float, default=0.3, help="share of meals held out as a test set (default 0.3)")
    p.add_argument("--table-weight", type=float, default=0.5,
                   help="training weight for meals whose nutrients come from the per-100 g sheet instead of a measurement (default 0.5)")
    p.add_argument("--max-side", type=int, default=1024)
    args = p.parse_args(argv)

    stats = ingest(args.meals, args.photos_dir, args.out, args.taxonomy, args.table, args.test_fraction, args.table_weight, args.max_side)
    print(f"{args.out / 'index.csv'}: {stats['meals']} meals, {stats['photos']} photos "
          f"({stats['test_meals']} meals held out as test), {stats['with_nutrition']} meals with calories/macros")
    if stats["with_nutrition"] == 0:
        print("no meal has calories or macros: these photos will train the mass output only until the per-100 g sheet is filled in")
    table = read_table(args.table, read_taxonomy(args.taxonomy))
    for w in energy_warnings(table):
        print("WARNING:", w)


if __name__ == "__main__":
    main()
