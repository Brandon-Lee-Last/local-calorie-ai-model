"""Turn folders of dish photos you collected yourself into a training folder (the ``--extra-data`` format).

    python -m calorie_model.ingest_folder --src my_downloads --out data/sa_web --source web

``my_downloads`` holds one sub-folder per dish, named with a dish id, name or alias from
``datasets/south_african/dishes.csv`` (``bunny_chow/``, ``Boerewors/``, ``vetkoek/`` ...), each full of photos.
How you collected them is up to you; this tool does not download anything.

* Every photo is re-encoded as an upright, metadata-free JPEG (no GPS position or camera serial is kept).
* Exact copies and near copies (same picture resized or re-saved) are dropped, so the same photo cannot end up
  on both sides of the train/test split. Photos by one photographer of one plate are *not* caught, so a test
  score on these photos flatters the model; trust the weighed-meal test instead.
* The photos carry **no nutrition labels**. Like the Commons photos they teach the food / not-food output and
  give examples per dish; they cannot teach calories.
* ``--source`` tags the folder in the report. Keep photos whose licence you cannot show in their own folder, apart
  from licensed ones, and leave that folder out of the ``--extra-data`` list for any model you ship: a model
  trained on them is not "private use" once it is inside an app other people install.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from PIL import Image, ImageOps

from .ingest_meals import is_test, save_clean_jpeg
from .sa_foods import DEFAULT_TAXONOMY, read_taxonomy, resolve_dish

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
NEAR_DUPLICATE_BITS = 4  # average hashes this close are the same picture


def average_hash(path: Path) -> int:
    """64-bit fingerprint: the picture as 8x8 grey squares, each above or below the mean. Survives resizing
    and re-saving, not cropping."""
    with Image.open(path) as im:
        small = ImageOps.exif_transpose(im).convert("L").resize((8, 8), Image.BILINEAR)
    pixels = list(small.tobytes())  # one byte per grey pixel
    mean = sum(pixels) / 64
    return sum(1 << i for i, p in enumerate(pixels) if p > mean)


def ingest(
    src: Path,
    out: Path,
    taxonomy_path: Path = DEFAULT_TAXONOMY,
    source: str = "web",
    test_fraction: float = 0.1,
    min_side: int = 160,
    max_side: int = 640,
) -> dict[str, int]:
    taxonomy = read_taxonomy(taxonomy_path)
    folders = sorted(d for d in src.iterdir() if d.is_dir())
    if not folders:
        raise ValueError(f"{src}: expected one sub-folder per dish")
    dish_of: dict[Path, str] = {}
    for d in folders:  # resolve every folder name first, so a typo stops the run before any work is done
        try:
            dish_of[d] = resolve_dish(d.name, taxonomy)
        except ValueError as e:
            raise ValueError(f"{src}: folder {d.name!r}: {e}") from None

    stats = {"photos": 0, "unreadable": 0, "too_small": 0, "duplicates": 0}
    seen_bytes: set[str] = set()
    rows: list[dict[str, str]] = []
    for folder, dish in dish_of.items():
        hashes: list[int] = []
        for path in sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUFFIXES):
            digest = hashlib.sha1(path.read_bytes()).hexdigest()
            if digest in seen_bytes:
                stats["duplicates"] += 1
                continue
            seen_bytes.add(digest)
            try:
                with Image.open(path) as im:
                    short_side = min(im.size)
                fingerprint = average_hash(path)
            except OSError:
                stats["unreadable"] += 1
                continue
            if short_side < min_side:
                stats["too_small"] += 1
                continue
            if any((fingerprint ^ h).bit_count() <= NEAR_DUPLICATE_BITS for h in hashes):
                stats["duplicates"] += 1
                continue
            hashes.append(fingerprint)
            rel = f"img/{dish}/{digest[:16]}.jpg"
            save_clean_jpeg(path, out / rel, max_side)
            rows.append({"path": rel, "split": "test" if is_test(digest, test_fraction) else "", "dish_id": dish, "source": source})
            stats["photos"] += 1

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["path", "split", "dish_id", "source"])
        w.writeheader()
        w.writerows(rows)
    stats["dishes"] = len({r["dish_id"] for r in rows})
    return stats


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="folder with one sub-folder of photos per dish")
    p.add_argument("--out", type=Path, required=True, help="output folder to pass to train --extra-data")
    p.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    p.add_argument("--source", default="web", help="label for the report; keep unlicensed photos under their own label and folder")
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--min-side", type=int, default=160, help="skip photos smaller than this on their short side")
    p.add_argument("--max-side", type=int, default=640)
    args = p.parse_args(argv)

    s = ingest(args.src, args.out, args.taxonomy, args.source, args.test_fraction, args.min_side, args.max_side)
    print(f"{args.out / 'index.csv'}: {s['photos']} photos, {s['dishes']} dishes "
          f"(dropped {s['duplicates']} duplicates, {s['too_small']} too small, {s['unreadable']} unreadable)")


if __name__ == "__main__":
    main()
