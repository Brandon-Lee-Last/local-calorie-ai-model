"""Download openly licensed photos of South African dishes from Wikimedia Commons.

    python -m calorie_model.prepare_commons download --out data/sa_commons --contact you@example.com
    # look through data/sa_commons/img/<dish>/ and delete every photo that is not that dish
    python -m calorie_model.prepare_commons index --out data/sa_commons

These photos have **no nutrition labels**. They teach the food / not-food output to accept South African food
(boerewors, pap and chakalaka should never be rejected as "not food") and give a pool of examples per dish
for a later dish-recognition head. They cannot teach calories; see ``docs/south-african-foods.md``.

* Only photos under a licence this tool recognises are kept: public domain / CC0 / CC BY by default,
  CC BY-SA with ``--allow-sharealike``. NonCommercial and NoDerivatives photos are never kept.
* Author, licence and the file's Commons page are recorded in ``attribution.csv`` for every photo. CC BY and
  CC BY-SA require crediting the author, so keep that file and show the credits in the app.
* Commons search is by text, so some hits will be the wrong thing (a menu, a map, a different dish). The
  ``index`` step only lists photos still on disk, so deleting a photo is how you reject it.
* Wikimedia asks automated clients to identify themselves: ``--contact`` (an email or URL) goes in the
  User-Agent. Requests are paced (``--delay``); do not turn that off.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .ingest_meals import is_test
from .sa_foods import DEFAULT_TAXONOMY, read_taxonomy

API = "https://commons.wikimedia.org/w/api.php"
ATTRIBUTION_COLUMNS = ["path", "dish_id", "title", "page_url", "author", "licence", "licence_url", "sha1", "query"]
MIME_EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
SOURCE = "commons"

FetchJson = Callable[[dict[str, str]], dict]
FetchFile = Callable[[str, Path], None]


def classify_licence(short_name: str) -> str | None:
    """``pd``, ``cc-by`` or ``cc-by-sa``, or None for anything else (NonCommercial, NoDerivatives, unknown)."""
    s = re.sub(r"[\s_]+", " ", short_name.strip().lower())
    if not s:
        return None
    if re.search(r"\b(nc|nd)\b", s) or "noncommercial" in s or "no derivatives" in s:
        return None
    if s.startswith("cc0") or s in ("pd", "pdm", "public domain") or s.startswith(("public domain", "pd ", "pd-")):
        return "pd"
    m = re.match(r"cc[ -]by(?:[ -](sa))?(?:[ -]\d|$)", s)
    if m:
        return "cc-by-sa" if m.group(1) else "cc-by"
    return None


def clean_html(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", text or ""))).strip()


@dataclass(frozen=True)
class Candidate:
    page_id: int
    rank: int
    title: str
    page_url: str
    file_url: str
    mime: str
    width: int
    height: int
    sha1: str
    author: str
    licence: str
    licence_class: str | None
    licence_url: str


def parse_candidates(payload: dict) -> list[Candidate]:
    """Candidates from one Commons ``generator=search`` + ``imageinfo`` response, in search-rank order."""
    out = []
    for page in (payload.get("query") or {}).get("pages", {}).values():
        info = (page.get("imageinfo") or [None])[0]
        if not info:
            continue
        meta = info.get("extmetadata") or {}

        def field(key: str) -> str:
            return clean_html((meta.get(key) or {}).get("value", ""))

        licence = field("LicenseShortName")
        out.append(Candidate(
            page_id=int(page["pageid"]),
            rank=int(page.get("index", 0)),
            title=page.get("title", ""),
            page_url=info.get("descriptionurl", ""),
            file_url=info.get("thumburl") or info.get("url", ""),  # a 640 px rendition, not the original
            mime=info.get("mime", ""),
            width=int(info.get("width") or 0),
            height=int(info.get("height") or 0),
            sha1=info.get("sha1", ""),
            author=field("Artist") or field("Credit"),
            licence=licence,
            licence_class=classify_licence(licence),
            licence_url=field("LicenseUrl"),
        ))
    return sorted(out, key=lambda c: c.rank)


def wanted(c: Candidate, allow_sharealike: bool, min_side: int) -> bool:
    allowed = {"pd", "cc-by"} | ({"cc-by-sa"} if allow_sharealike else set())
    return (
        c.licence_class in allowed
        and c.mime in MIME_EXT
        and bool(c.file_url)
        and bool(c.author or c.licence_class == "pd")  # a credit is required for everything but public domain
        and min(c.width, c.height) >= min_side
    )


def http_json(user_agent: str, delay: float) -> FetchJson:
    def fetch(params: dict[str, str]) -> dict:
        url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
        return json.loads(_get(url, user_agent, delay))
    return fetch


def http_file(user_agent: str, delay: float) -> FetchFile:
    def fetch(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_get(url, user_agent, delay))
    return fetch


def _get(url: str, user_agent: str, delay: float, tries: int = 5) -> bytes:
    for attempt in range(tries):
        time.sleep(delay)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": user_agent})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
            time.sleep(delay * 2 ** (attempt + 2))  # back off harder each time
        except (urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
            time.sleep(delay * 2 ** (attempt + 2))
    raise AssertionError("unreachable")


def search(fetch: FetchJson, term: str, limit: int) -> list[Candidate]:
    """Up to ``limit`` Commons bitmap files matching ``term``, following result pages."""
    params = {
        "action": "query", "generator": "search", "gsrsearch": f"{term} filetype:bitmap", "gsrnamespace": "6",
        "gsrlimit": str(min(50, limit)), "prop": "imageinfo", "iiprop": "url|size|mime|sha1|extmetadata",
        "iiurlwidth": "640", "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|Credit",
    }
    found: list[Candidate] = []
    while len(found) < limit:
        payload = fetch(params)
        found += parse_candidates(payload)
        more = payload.get("continue")
        if not more:
            break
        params = {**params, **more}
    return found[:limit]


def download(
    out: Path,
    taxonomy_path: Path,
    fetch: FetchJson,
    fetch_file: FetchFile,
    per_dish: int = 40,
    per_query: int = 60,
    allow_sharealike: bool = False,
    min_side: int = 300,
    only: set[str] | None = None,
) -> dict[str, int]:
    taxonomy = read_taxonomy(taxonomy_path)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "attribution.csv"
    known = _read_attribution(log)
    seen_sha = {r["sha1"] for r in known.values()}  # includes photos you deleted, so a rejected photo stays rejected
    counts = {d: sum(1 for r in known.values() if r["dish_id"] == d and (out / r["path"]).is_file()) for d in taxonomy}
    new = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, ATTRIBUTION_COLUMNS)
        if new:
            writer.writeheader()
        for dish in taxonomy.values():
            if only and dish.dish_id not in only:
                continue
            for term in dish.queries:
                if counts[dish.dish_id] >= per_dish:
                    break
                for c in search(fetch, term, per_query):
                    if counts[dish.dish_id] >= per_dish:
                        break
                    if not wanted(c, allow_sharealike, min_side) or c.sha1 in seen_sha:
                        continue
                    rel = f"img/{dish.dish_id}/{c.page_id}{MIME_EXT[c.mime]}"
                    fetch_file(c.file_url, out / rel)
                    seen_sha.add(c.sha1)
                    counts[dish.dish_id] += 1
                    row = {"path": rel, "dish_id": dish.dish_id, "title": c.title, "page_url": c.page_url, "author": c.author,
                           "licence": c.licence, "licence_url": c.licence_url, "sha1": c.sha1, "query": term}
                    writer.writerow(row)
                    f.flush()
    return counts


def _read_attribution(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["path"]: r for r in csv.DictReader(f)}


def build_index(out: Path, test_fraction: float = 0.15) -> dict[str, int]:
    """``index.csv`` (the ``--extra-data`` format, no nutrition) for the photos still on disk.

    Test photos are chosen by *author*, so one photographer's series of the same dish cannot sit on both sides
    and flatter the held-out number.
    """
    rows = _read_attribution(out / "attribution.csv")
    if not rows:
        raise FileNotFoundError(f"{out / 'attribution.csv'} is missing or empty; run the download step first")
    kept = [r for r in rows.values() if (out / r["path"]).is_file()]
    per_dish: dict[str, int] = {}
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["path", "split", "dish_id", "source"])
        w.writeheader()
        for r in kept:
            author_key = r["author"] or hashlib.sha1(r["path"].encode()).hexdigest()
            w.writerow({"path": r["path"], "split": "test" if is_test(author_key, test_fraction) else "", "dish_id": r["dish_id"], "source": SOURCE})
            per_dish[r["dish_id"]] = per_dish.get(r["dish_id"], 0) + 1
    return per_dish


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    d = sub.add_parser("download", help="search Commons and download openly licensed photos")
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--contact", required=True, help="your email or a URL; sent in the User-Agent as Wikimedia asks")
    d.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    d.add_argument("--dish", action="append", default=None, metavar="DISH_ID", help="only these dishes (repeatable)")
    d.add_argument("--per-dish", type=int, default=40)
    d.add_argument("--per-query", type=int, default=60, help="search hits to look through per search term")
    d.add_argument("--allow-sharealike", action="store_true",
                   help="also keep CC BY-SA photos (check first what share-alike means for a trained model)")
    d.add_argument("--min-side", type=int, default=300, help="skip photos smaller than this on their short side")
    d.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    i = sub.add_parser("index", help="write index.csv for the photos still on disk")
    i.add_argument("--out", type=Path, required=True)
    i.add_argument("--test-fraction", type=float, default=0.15)
    args = p.parse_args(argv)

    if args.step == "download":
        agent = f"calorie-model-sa-data/0.1 ({args.contact}) python-urllib"
        counts = download(args.out, args.taxonomy, http_json(agent, args.delay), http_file(agent, args.delay),
                          args.per_dish, args.per_query, args.allow_sharealike, args.min_side, set(args.dish) if args.dish else None)
        for dish_id, n in sorted(counts.items(), key=lambda kv: kv[1]):
            print(f"{dish_id:24s} {n:3d}")
        print(f"{sum(counts.values())} photos under {args.out / 'img'}; review them, then run the index step")
    else:
        per_dish = build_index(args.out, args.test_fraction)
        print(f"{args.out / 'index.csv'}: {sum(per_dish.values())} photos, {len(per_dish)} dishes")


if __name__ == "__main__":
    main()
