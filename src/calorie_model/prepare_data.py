"""Download Nutrition5k from its public bucket and build a compact local image cache.

    python -m calorie_model.prepare_data --out data/nutrition5k                   # overhead + side-angle frames
    python -m calorie_model.prepare_data --out data/nutrition5k --views overhead  # overhead photos only (~1.5 GB)
    python -m calorie_model.prepare_data --out data/nutrition5k --limit 20        # quick test run

Plain HTTPS from the public bucket, so no gsutil or Google account is needed; it
needs ``pillow`` and ``av`` (``pip install av``). Safe to interrupt and re-run:
files that already exist are skipped.

What it fetches, per labelled dish:

* ``overhead``: the overhead ``rgb.png`` (only ~3.5k of the ~5k dishes have one).
* ``side``: the side-angle videos are raw H.264, ~20 MB each (190 GB in total), so
  only the first ``--side-bytes`` of each camera is requested and the first frame
  is decoded. The last frame of a cut stream is partial, so it is dropped. Many
  dishes are missing some or all of the four cameras.

Every image is squashed to ``--size`` x ``--size`` (the same squash the app does)
and stored as JPEG. Output layout::

    <out>/metadata/dish_metadata_cafe{1,2}.csv   raw labels
    <out>/splits/rgb_{train,test}_ids.txt        official splits
    <out>/labels.csv                             dish_id, calories, mass, fat, carb, protein
    <out>/cache/<dish_id>/overhead.jpg, side_<cam>_<k>.jpg
    <out>/index.csv                              dish_id, split, view, path

Dishes in no official split but with labels are added to ``train`` as extra data
(disable with ``--no-extra``). The official test split is never touched.
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

BASE = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/"
CAMERAS = "ABCD"
LABEL_COLUMNS = ("calories", "mass", "fat", "carb", "protein")


def fetch(url: str, byte_limit: int | None = None, retries: int = 4) -> bytes | None:
    """GET ``url`` (only the first ``byte_limit`` bytes if given); ``None`` if it does not exist."""
    headers = {"Range": f"bytes=0-{byte_limit - 1}"} if byte_limit else {}
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                return r.read(byte_limit) if byte_limit else r.read()
        except urllib.error.HTTPError as e:
            if e.code in (404, 416):  # 416: asked for bytes of an empty file
                return None
            if attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries:
                raise
        time.sleep(2**attempt)
    return None  # unreachable


def decode_frames(data: bytes, truncated: bool, count: int = 1) -> list[Image.Image]:
    """Decode up to ``count`` evenly spaced complete frames from a raw H.264 stream."""
    import av

    frames = []
    try:
        with av.open(io.BytesIO(data), format="h264") as container:
            frames = list(container.decode(video=0))
    except av.error.FFmpegError:
        pass  # a cut stream can fail at the tail; keep what decoded before it
    if truncated and frames:
        frames.pop()  # the final frame of a cut stream is partial (a green band)
    if not frames:
        return []
    idx = sorted({round(i * (len(frames) - 1) / max(count - 1, 1)) for i in range(count)})
    return [frames[i].to_image() for i in idx]


def save_square(img: Image.Image, path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").resize((size, size), Image.BILINEAR).save(path, quality=95)


def process_dish(dish: str, out: Path, views: set[str], size: int, side_bytes: int, side_frames: int) -> None:
    folder = out / "cache" / dish
    if "overhead" in views and not (folder / "overhead.jpg").exists():
        data = fetch(f"{BASE}imagery/realsense_overhead/{dish}/rgb.png")
        if data:
            with Image.open(io.BytesIO(data)) as im:
                save_square(im, folder / "overhead.jpg", size)
    if "side" in views:
        for cam in CAMERAS:
            if any(folder.glob(f"side_{cam}_*.jpg")):
                continue
            limit = side_bytes
            for _ in range(3):  # widen the range if too few frames decoded
                data = fetch(f"{BASE}imagery/side_angles/{dish}/camera_{cam}.h264", byte_limit=limit)
                if data is None:
                    break
                truncated = len(data) >= limit
                frames = decode_frames(data, truncated, side_frames)
                if frames or not truncated:
                    for k, frame in enumerate(frames):
                        save_square(frame, folder / f"side_{cam}_{k}.jpg", size)
                    break
                limit *= 2


def parse_metadata(text: str) -> dict[str, list[float]]:
    """Dish-level labels from a metadata CSV.

    Rows are headerless: ``dish_id, calories, mass, fat, carb, protein`` then a
    variable number of per-ingredient fields, which are ignored here.
    """
    labels = {}
    for row in csv.reader(io.StringIO(text)):
        if len(row) >= 6:
            labels[row[0]] = [float(v) for v in row[1:6]]
    return labels


def build_index(out: Path, split_of: dict[str, str]) -> int:
    """(Re)write ``index.csv`` from the images present in the cache; returns the row count."""
    rows = []
    for dish, split in sorted(split_of.items()):
        for img in sorted((out / "cache" / dish).glob("*.jpg")):
            view = "overhead" if img.name.startswith("overhead") else "side"
            rows.append((dish, split, view, f"cache/{dish}/{img.name}"))
    with open(out / "index.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("dish_id", "split", "view", "path"))
        writer.writerows(rows)
    return len(rows)


def read_ids(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def prepare(
    out: Path,
    views: set[str],
    size: int = 256,
    side_bytes: int = 1_000_000,
    side_frames: int = 1,
    workers: int = 16,
    limit: int | None = None,
    extra: bool = True,
) -> None:
    (out / "metadata").mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(parents=True, exist_ok=True)

    labels: dict[str, list[float]] = {}
    for cafe in (1, 2):
        text = fetch(f"{BASE}metadata/dish_metadata_cafe{cafe}.csv")
        if text is None:
            raise RuntimeError(f"dish_metadata_cafe{cafe}.csv not found in the bucket")
        (out / "metadata" / f"dish_metadata_cafe{cafe}.csv").write_bytes(text)
        labels.update(parse_metadata(text.decode()))

    split_of: dict[str, str] = {}
    for split in ("train", "test"):
        text = fetch(f"{BASE}dish_ids/splits/rgb_{split}_ids.txt")
        if text is None:
            raise RuntimeError(f"rgb_{split}_ids.txt not found in the bucket")
        (out / "splits" / f"rgb_{split}_ids.txt").write_bytes(text)
        split_of.update({d: split for d in read_ids(text.decode()) if d in labels})
    if extra:
        split_of.update({d: "train" for d in labels if d not in split_of})

    with open(out / "labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("dish_id", *LABEL_COLUMNS))
        writer.writerows((d, *labels[d]) for d in sorted(split_of))

    dishes = sorted(split_of)
    if limit:
        dishes = random.Random(0).sample(dishes, min(limit, len(dishes)))
        split_of = {d: split_of[d] for d in dishes}
    counts = {s: sum(v == s for v in split_of.values()) for s in ("train", "test")}
    print(f"{len(dishes)} labelled dishes ({counts['train']} train, {counts['test']} test); views: {sorted(views)}")

    failures, done, start = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_dish, d, out, views, size, side_bytes, side_frames): d for d in dishes}
        for fut in as_completed(futures):
            done += 1
            try:
                fut.result()
            except Exception as e:  # keep going; report at the end and re-run to retry
                failures.append((futures[fut], f"{type(e).__name__}: {e}"))
            if done % 100 == 0 or done == len(dishes):
                print(f"  {done}/{len(dishes)} dishes, {time.time() - start:.0f}s, {len(failures)} failed")

    n = build_index(out, split_of)
    per_view = {v: 0 for v in ("overhead", "side")}
    with open(out / "index.csv") as f:
        for row in csv.DictReader(f):
            per_view[row["view"]] += 1
    print(f"index.csv: {n} images ({per_view['overhead']} overhead, {per_view['side']} side)")
    if failures:
        (out / "failures.txt").write_text("\n".join(f"{d}\t{msg}" for d, msg in failures))
        print(f"{len(failures)} dishes failed (see {out / 'failures.txt'}); re-run to retry them")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/nutrition5k"))
    p.add_argument("--views", default="overhead,side", help="comma-separated: overhead, side")
    p.add_argument("--size", type=int, default=256, help="cache images are squashed to size x size")
    p.add_argument("--side-bytes", type=int, default=1_000_000, help="bytes fetched from each side video")
    p.add_argument("--side-frames", type=int, default=1, help="frames kept per side camera")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="only process a random sample of N dishes")
    p.add_argument("--no-extra", action="store_true", help="do not add unsplit labelled dishes to train")
    args = p.parse_args(argv)
    views = set(args.views.split(","))
    if not views <= {"overhead", "side"}:
        p.error("--views must be a comma-separated subset of: overhead, side")
    prepare(args.out, views, args.size, args.side_bytes, args.side_frames, args.workers, args.limit, not args.no_extra)


if __name__ == "__main__":
    main()
