"""Build a food / not-food image cache from Open Images, to teach the model what is NOT food.

    python -m calorie_model.prepare_openimages --out data/openimages
    python -m calorie_model.prepare_openimages --out data/openimages --scale 0.02   # tiny test run

Why: a model trained only on food photos has no idea what a non-food photo is, so a photo of a dog
still gets five confident-looking nutrition numbers. This adds real-world food photos (positives) and
many kinds of non-food photos, including animals and people (negatives).

Source: Open Images (Google). The images are listed as CC BY 2.0 with the author recorded for each
one; the annotations are CC BY 4.0. ``index.csv`` keeps author, licence and landing page for every
image used, so attribution can be produced. Whether training a model on CC BY images needs more than
that is a question for someone qualified; this is not legal advice.

Labels are human-verified but not exhaustive: an image without a food label might still show food.
To keep the negatives clean, only images that have at least one verified label and no food or
food-adjacent label (plates, bowls, drinks, kitchens...) count as negatives; images with food-adjacent
labels but no food label are left out of both groups.

Images come from the public S3 mirror of Open Images, resized (squashed, like the app) to ``--size``
and stored as JPEG. Safe to interrupt and re-run. Layout::

    <out>/meta/*.csv                          downloaded label and image-metadata files
    <out>/cache/<subset>/<image_id>.jpg
    <out>/index.csv                           image_id, split, label, group, path, subset, license,
                                              author, landing_url, title

``split`` is ``train`` for images from the Open Images *test* subset and ``test`` for the Open Images
*validation* subset, so the held-out set never overlaps the training pool.
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from .prepare_data import fetch, save_square

META_BASE = "https://storage.googleapis.com/openimages/"
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{subset}/{image_id}.jpg"

# Open Images subset -> our split. The held-out split never overlaps the training pool.
SPLITS = {"test": "train", "validation": "test"}
LABEL_FILES = {
    "test": "v5/test-annotations-human-imagelabels.csv",
    "validation": "v5/validation-annotations-human-imagelabels.csv",
}
IMAGE_FILES = {
    "test": "2018_04/test/test-images-with-rotation.csv",
    "validation": "2018_04/validation/validation-images-with-rotation.csv",
}

# Class IDs from oidv6-class-descriptions.csv.
FOOD = {
    "/m/02wbm", "/m/02q08p0", "/m/0krfg", "/m/01_bhs", "/m/052lwg6", "/m/01ww8y", "/m/0270h",
    "/m/06nwz", "/m/0663v", "/m/0cdn1", "/m/0l515", "/m/0fszt", "/m/021mn", "/m/0grw1",
    "/m/0hz4q", "/m/0jfd5", "/m/0jffp", "/m/01ykh", "/m/02xwb", "/m/0f4s2w",
}  # fmt: skip  (Food, Dish, Meal, Fast food, Baked goods, Snack, Dessert, Seafood, Pizza, Hamburger,
#   Sandwich, Cake, Cookie, Salad, Breakfast, Lunch, Dinner, Cuisine, Fruit, Vegetable)
FOOD_ADJACENT = {"/m/0271t", "/m/04brg2", "/m/050gv4", "/m/04kkgm", "/m/0d4wf"}  # Drink, Tableware, Plate, Bowl, Kitchen
ANIMALS = {"/m/0bt9lr", "/m/01yrx"}  # Dog, Cat
PEOPLE = {"/m/01g317"}  # Person

# Images per group at --scale 1: (training pool, held-out).
TARGETS = {
    "food": (8000, 1000),
    "animal": (2500, 500),
    "person": (3000, 500),
    "other": (6500, 1000),
}
GROUP_LABEL = {"food": "food", "animal": "nonfood", "person": "nonfood", "other": "nonfood"}


def classify(positive_labels: dict[str, set[str]]) -> dict[str, str]:
    """image_id -> group (food / animal / person / other); ambiguous images are left out."""
    groups = {}
    for image_id, labels in positive_labels.items():
        if not labels:
            continue  # never looked at by a labeller: could be anything
        if labels & FOOD:
            groups[image_id] = "food"
        elif labels & FOOD_ADJACENT:
            continue
        elif labels & ANIMALS:
            groups[image_id] = "animal"
        elif labels & PEOPLE:
            groups[image_id] = "person"
        else:
            groups[image_id] = "other"
    return groups


def select(groups: dict[str, str], counts: dict[str, int], seed: int = 0) -> dict[str, str]:
    """Deterministically sample up to ``counts[group]`` image ids per group."""
    rng = random.Random(seed)
    chosen = {}
    for group, n in counts.items():
        ids = sorted(i for i, g in groups.items() if g == group)
        for image_id in rng.sample(ids, min(n, len(ids))):
            chosen[image_id] = group
    return chosen


def read_positive_labels(text: str) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = defaultdict(set)
    for row in csv.DictReader(io.StringIO(text)):
        labels[row["ImageID"]]  # an image with only negative labels is still "seen"
        if row["Confidence"] == "1":
            labels[row["ImageID"]].add(row["LabelName"])
    return labels


def read_image_metadata(text: str, wanted: set[str]) -> dict[str, dict[str, str]]:
    meta = {}
    for row in csv.DictReader(io.StringIO(text)):
        if row["ImageID"] in wanted:
            meta[row["ImageID"]] = row
    return meta


def cached_text(url: str, dest: Path) -> str:
    if not dest.exists():
        data = fetch(url)
        if data is None:
            raise RuntimeError(f"not found: {url}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return dest.read_text(encoding="utf-8", errors="replace")


def download_image(subset: str, image_id: str, out: Path, size: int) -> bool:
    dest = out / "cache" / subset / f"{image_id}.jpg"
    if dest.exists():
        return True
    data = fetch(IMAGE_URL.format(subset=subset, image_id=image_id))
    if not data:
        return False
    with Image.open(io.BytesIO(data)) as im:
        save_square(im, dest, size)
    return True


def prepare(out: Path, scale: float = 1.0, size: int = 256, seed: int = 0, workers: int = 16) -> None:
    rows, failures = [], []
    for subset, split in SPLITS.items():
        pool_index = 0 if split == "train" else 1
        counts = {g: max(1, round(n[pool_index] * scale)) for g, n in TARGETS.items()}
        labels = read_positive_labels(cached_text(META_BASE + LABEL_FILES[subset], out / "meta" / f"{subset}_labels.csv"))
        chosen = select(classify(labels), counts, seed)
        meta = read_image_metadata(
            cached_text(META_BASE + IMAGE_FILES[subset], out / "meta" / f"{subset}_images.csv"), set(chosen)
        )
        print(f"{subset} -> {split}: {len(chosen)} images ({', '.join(f'{g} {sum(v == g for v in chosen.values())}' for g in counts)})")

        done, start = 0, time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(download_image, subset, i, out, size): i for i in chosen}
            for fut in as_completed(futures):
                image_id = futures[fut]
                done += 1
                try:
                    ok = fut.result()
                except Exception as e:  # keep going; re-run to retry
                    ok = False
                    failures.append((image_id, f"{type(e).__name__}: {e}"))
                if ok:
                    m = meta.get(image_id, {})
                    group = chosen[image_id]
                    rows.append(
                        (image_id, split, GROUP_LABEL[group], group, f"cache/{subset}/{image_id}.jpg", subset,
                         m.get("License", ""), m.get("Author", ""), m.get("OriginalLandingURL", ""), m.get("Title", ""))
                    )  # fmt: skip
                if done % 500 == 0 or done == len(chosen):
                    print(f"  {done}/{len(chosen)}, {time.time() - start:.0f}s, {len(failures)} failed")

    rows.sort()
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("image_id", "split", "label", "group", "path", "subset", "license", "author", "landing_url", "title"))
        writer.writerows(rows)
    summary = defaultdict(int)
    for r in rows:
        summary[(r[1], r[3])] += 1
    print("index.csv:", len(rows), "images", dict(sorted(summary.items())))
    if failures:
        (out / "failures.txt").write_text("\n".join(f"{i}\t{m}" for i, m in failures))
        print(f"{len(failures)} failed (see {out / 'failures.txt'}); re-run to retry")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/openimages"))
    p.add_argument("--scale", type=float, default=1.0, help="multiplier on the image counts (0.02 = quick test)")
    p.add_argument("--size", type=int, default=256, help="cache images are squashed to size x size")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args(argv)
    prepare(args.out, args.scale, args.size, args.seed, args.workers)


if __name__ == "__main__":
    main()
