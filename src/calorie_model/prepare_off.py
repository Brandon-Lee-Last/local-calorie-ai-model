"""List the South African products in Open Food Facts, with their nutrition, ready to download as a CSV.

    python -m calorie_model.prepare_off --out data/off_za --contact you@example.com
    python -m calorie_model.prepare_off --out data/off_za --contact you@example.com --images 2000

Open Food Facts is a free, open database of packaged foods that people photograph and enter. It is the lawful way to get
South African packaged foods (juices, sweets, chocolate, biscuits, cereal, snacks...) with per-100 g nutrition, a barcode and a
front-of-pack photo. This reads their full export once, streaming (nothing big is saved), and keeps the South African rows.

What you get, and what you do not:
* ``products.csv``: barcode, name, brand, group, nutrition per 100 g, serving size, photo link, and the page to credit.
* with ``--images N``: up to N front-of-pack photos, politely paced, with author and licence noted.
* These are **pack photos**. They can feed a packaged-food recogniser and the per-100 g sheet. They cannot teach the plate model
  how big a portion is, so they are NOT in the calorie training format and should not be mixed into it.

Licences: the database is ODbL (credit "Open Food Facts contributors", and share changes to the database under the same
licence); the photos are CC BY-SA. Ask someone qualified what that means for a commercial product before you ship anything
built on them. Open Food Facts asks for a descriptive User-Agent (``--contact``) and gentle use.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import time
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator

DUMP_URL = "https://static.openfoodfacts.org/data/openfoodfacts-products.jsonl.gz"
COUNTRY_TAG = "en:south-africa"
COLUMNS = [
    "barcode", "name", "brand", "group", "categories", "quantity", "serving_size", "serving_g", "calories_100g", "protein_100g",
    "carb_100g", "fat_100g", "sugar_100g", "fiber_100g", "sodium_mg_100g", "image_url", "page_url", "image_path", "source", "licence",
]  # fmt: skip

# First match wins. Checked against Open Food Facts category tags (e.g. "en:fruit-juices").
GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("juice", ("juice", "nectar")),
    ("soft_drink", ("soda", "carbonated", "soft-drink", "cola", "energy-drink", "sweetened-beverages", "iced-tea")),
    ("water", ("waters",)),
    ("dairy_drink", ("milk-drink", "flavoured-milk", "drinkable-yogurt", "maas", "amasi")),
    ("chocolate", ("chocolate", "cocoa")),
    ("sweets", ("candies", "confectioneries", "sweets", "gummies", "lollipop", "marshmallow", "chewing-gum", "toffee")),
    ("biscuits", ("biscuits", "cookies", "rusks", "wafers")),
    ("cake_dessert", ("cakes", "desserts", "puddings", "ice-cream", "pastries")),
    ("snacks", ("snacks", "chips", "crisps", "popcorn", "nuts", "biltong", "jerky", "dried-meat")),
    ("cereal", ("cereals", "breakfast", "porridge", "oats", "muesli")),
    ("bread_bakery", ("breads", "bakery")),
    ("meat", ("meats", "sausages", "boerewors", "polony", "poultry", "fish")),
    ("sauce_spread", ("sauces", "spreads", "condiments", "jams", "chutney", "peanut-butter")),
    ("meal", ("meals", "ready-meals", "pizzas", "soups", "pasta", "rice", "noodles")),
    ("dairy", ("dairies", "yogurts", "cheeses", "milks", "butter")),
    ("fruit_veg", ("fruits", "vegetables", "legumes")),
]  # fmt: skip


def _num(v: object) -> float | None:
    try:
        n = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return n if n == n and n >= 0 else None


def group_of(categories: Iterable[str]) -> str:
    """Whole-word match on the category tags: "cola" must not match inside "chocolates"."""
    padded = ["-" + c.lower().split(":", 1)[-1] + "-" for c in categories]
    for name, needles in GROUPS:
        if any(f"-{n}-" in t or f"-{n}s-" in t for t in padded for n in needles):
            return name
    return "other"


def parse_product(p: dict) -> dict | None:
    """One Open Food Facts product -> one row, or None if it is not South African or has no usable energy figure."""
    if COUNTRY_TAG not in (p.get("countries_tags") or []):
        return None
    n = p.get("nutriments") or {}
    kcal = _num(n.get("energy-kcal_100g"))
    if kcal is None:
        kj = _num(n.get("energy_100g"))
        kcal = kj / 4.184 if kj is not None else None
    name = next((str(p[k]).strip() for k in ("product_name_en", "product_name", "generic_name") if str(p.get(k) or "").strip()), "")
    code = str(p.get("code") or p.get("_id") or "").strip()
    if kcal is None or kcal > 950 or not name or not code:
        return None  # nothing beats pure fat at about 900 kcal per 100 g, so more is a data error
    sodium = _num(n.get("sodium_100g"))
    serving = _num(p.get("serving_quantity"))
    cats = p.get("categories_tags") or []
    r = lambda x: "" if x is None else round(x, 2)  # noqa: E731
    return {
        "barcode": code,
        "name": name[:200],
        "brand": str(p.get("brands") or "").split(",")[0].strip()[:100],
        "group": group_of(cats),
        "categories": "|".join(c.removeprefix("en:") for c in cats[:6]),
        "quantity": str(p.get("quantity") or "")[:60],
        "serving_size": str(p.get("serving_size") or "")[:60],
        "serving_g": r(serving) if serving and 0 < serving <= 5000 else "",
        "calories_100g": r(kcal),
        "protein_100g": r(_num(n.get("proteins_100g")) or 0),
        "carb_100g": r(_num(n.get("carbohydrates_100g")) or 0),
        "fat_100g": r(_num(n.get("fat_100g")) or 0),
        "sugar_100g": r(_num(n.get("sugars_100g"))),
        "fiber_100g": r(_num(n.get("fiber_100g"))),
        "sodium_mg_100g": r(sodium * 1000) if sodium is not None else "",
        "image_url": str(p.get("image_front_url") or p.get("image_url") or ""),
        "page_url": f"https://world.openfoodfacts.org/product/{code}",
        "image_path": "",
        "source": "Open Food Facts contributors",
        "licence": "Data: ODbL. Photos: CC BY-SA",
    }  # fmt: skip


def open_source(source: str, contact: str | None) -> io.BufferedReader | gzip.GzipFile:
    """A local ``.jsonl`` / ``.jsonl.gz`` file, or the full export streamed from the web."""
    if source.startswith("http"):
        req = urllib.request.Request(source, headers={"User-Agent": f"calorie-model-research/0.1 ({contact})"})
        return gzip.GzipFile(fileobj=urllib.request.urlopen(req, timeout=120))  # type: ignore[arg-type]
    path = Path(source)
    return gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb")  # type: ignore[return-value]


def iter_south_african(lines: Iterable[bytes]) -> Iterator[dict]:
    for line in lines:
        line = line.strip()
        if not line or COUNTRY_TAG.encode() not in line:  # cheap test first: only parse the lines that can match
            continue
        try:
            row = parse_product(json.loads(line))
        except json.JSONDecodeError:
            continue
        if row:
            yield row


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(rows)


def download_images(rows: list[dict], out: Path, contact: str, limit: int, delay: float = 1.0, fetch=None) -> int:
    """Fetch up to ``limit`` front-of-pack photos for rows that have a link. Paced; skips what is already on disk."""
    fetch = fetch or (lambda url: urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": f"calorie-model-research/0.1 ({contact})"}), timeout=60).read())
    got = 0
    for row in rows:
        if got >= limit:
            break
        if not row["image_url"]:
            continue
        dest = out / "img" / row["group"] / f"{row['barcode']}.jpg"
        if dest.exists():
            row["image_path"] = f"img/{row['group']}/{dest.name}"
            continue
        try:
            data = fetch(row["image_url"])
        except Exception:  # one bad link should not stop the run
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        row["image_path"] = f"img/{row['group']}/{dest.name}"
        got += 1
        time.sleep(delay)
    return got


def summarize(rows: list[dict]) -> str:
    by: dict[str, int] = {}
    for r in rows:
        by[r["group"]] = by.get(r["group"], 0) + 1
    return ", ".join(f"{g} {n}" for g, n in sorted(by.items(), key=lambda kv: -kv[1]))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--contact", required=True, help="your email or URL, sent in the User-Agent as Open Food Facts asks")
    p.add_argument("--source", default=DUMP_URL, help="the export to read: a URL (default) or a downloaded .jsonl.gz")
    p.add_argument("--images", type=int, default=0, metavar="N", help="also download up to N front-of-pack photos")
    p.add_argument("--delay", type=float, default=1.0, help="seconds between photo downloads")
    args = p.parse_args(argv)

    print(f"reading {args.source} (the full export is several GB; only South African rows are kept)")
    rows = list(iter_south_african(open_source(args.source, args.contact)))
    print(f"{len(rows)} South African products with a usable energy figure: {summarize(rows)}")
    if args.images:
        n = download_images(rows, args.out, args.contact, args.images, args.delay)
        print(f"downloaded {n} photos")
    write_csv(rows, args.out / "products.csv")
    print(f"wrote {args.out / 'products.csv'}")


if __name__ == "__main__":
    main()
