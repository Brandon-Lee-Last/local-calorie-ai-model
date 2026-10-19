import csv
import io
import json
import math

import pytest
import torch
from PIL import Image

from calorie_model import prepare_commons as pc
from calorie_model.data import KNOWN_START, N_OUT, TARGET_WIDTH, ExtraNutritionDataset, load_rgb_uint8
from calorie_model.ingest_meals import ingest, is_test
from calorie_model.sa_foods import (
    DEFAULT_TABLE,
    DEFAULT_TAXONOMY,
    energy_warnings,
    meal_totals,
    parse_components,
    read_table,
    read_taxonomy,
    resolve_dish,
)
from calorie_model.train import main as train_main
from test_pipeline import IMG, _fake_cache, _fake_openimages


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


@pytest.fixture
def taxonomy(tmp_path):
    path = write_csv(tmp_path / "dishes.csv", ["dish_id", "name", "aliases", "category", "description", "commons_queries"], [
        ["pap", "Pap", "mielie pap|stywe pap", "starch", "", "pap maize porridge;stywe pap"],
        ["boerewors", "Boerewors", "wors", "meat", "", "boerewors"],
        ["chakalaka", "Chakalaka", "", "side", "", "chakalaka"],
    ])  # fmt: skip
    return path, read_taxonomy(path)


def table_file(tmp_path, rows):
    return write_csv(tmp_path / "table.csv", ["dish_id", "calories", "fat", "carb", "protein", "source", "source_ref"], rows)


# --- the shipped files ---------------------------------------------------------------------------------------


def test_shipped_dish_list_and_sheet_are_consistent():
    taxonomy = read_taxonomy(DEFAULT_TAXONOMY)
    assert len(taxonomy) >= 50 and all(d.queries for d in taxonomy.values())
    with open(DEFAULT_TABLE, newline="", encoding="utf-8") as f:
        sheet_ids = [r["dish_id"] for r in csv.DictReader(f)]
    assert sheet_ids == list(taxonomy)  # one row per dish, same order, so nothing is forgotten
    table = read_table(DEFAULT_TABLE, taxonomy)  # every filled row must name a source
    assert energy_warnings(table) == []


# --- nutrient sheet ------------------------------------------------------------------------------------------


def test_sheet_rejects_values_without_a_source_negatives_and_unknown_dishes(tmp_path, taxonomy):
    _, tax = taxonomy
    with pytest.raises(ValueError, match="no source"):
        read_table(table_file(tmp_path, [["pap", "100", "1", "20", "3", "", ""]]), tax)
    with pytest.raises(ValueError, match="non-negative"):
        read_table(table_file(tmp_path, [["pap", "-100", "1", "20", "3", "x", ""]]), tax)
    with pytest.raises(ValueError, match="not a number"):
        read_table(table_file(tmp_path, [["pap", "lots", "1", "20", "3", "x", ""]]), tax)
    with pytest.raises(ValueError, match="not in the dish list"):
        read_table(table_file(tmp_path, [["pizza", "100", "1", "20", "3", "x", ""]]), tax)
    assert read_table(table_file(tmp_path, [["pap", "", "", "", "", "", ""]]), tax) == {}  # a blank row is fine


def test_energy_check_flags_a_row_that_disagrees_with_its_own_macros(tmp_path, taxonomy):
    _, tax = taxonomy
    ok = read_table(table_file(tmp_path, [["pap", "116", "0.5", "25", "2.5", "x", ""]]), tax)  # 4*2.5+4*25+9*0.5 = 114.5
    bad = read_table(table_file(tmp_path, [["pap", "1160", "0.5", "25", "2.5", "x", ""]]), tax)
    assert energy_warnings(ok) == [] and "pap" in energy_warnings(bad)[0]


# --- components ----------------------------------------------------------------------------------------------


def test_components_accept_ids_names_and_aliases_and_suggest_on_typos(taxonomy):
    _, tax = taxonomy
    assert parse_components("pap:250|Wors:150|stywe pap:10", tax) == [("pap", 250.0), ("boerewors", 150.0), ("pap", 10.0)]
    with pytest.raises(ValueError, match="did you mean boerewors"):
        resolve_dish("boerwors", tax)
    for bad in ("pap", "pap:abc", "pap:0", "pap:-5", "pap:99999", ""):
        with pytest.raises(ValueError):
            parse_components(bad, tax)


def test_meal_totals_are_summed_and_left_unknown_when_a_component_lacks_a_value(tmp_path, taxonomy):
    _, tax = taxonomy
    table = read_table(table_file(tmp_path, [["pap", "100", "1", "20", "3", "x", ""], ["boerewors", "300", "25", "2", "15", "x", ""]]), tax)
    full = meal_totals(parse_components("pap:200|boerewors:100", tax), table)
    assert full == {"mass": 300.0, "calories": 500.0, "fat": 27.0, "carb": 42.0, "protein": 21.0}
    partial = meal_totals(parse_components("pap:200|chakalaka:50", tax), table)  # chakalaka has no row
    assert partial["mass"] == 250.0 and all(math.isnan(partial[n]) for n in ("calories", "fat", "carb", "protein"))


# --- weighed-meal ingest -------------------------------------------------------------------------------------


def make_photo(path, size=(80, 40), orientation=None):
    im = Image.new("RGB", size, (200, 100, 50))
    if orientation:
        exif = Image.Exif()
        exif[0x0112] = orientation
        exif[0x010F] = "secret-camera-make"
        im.save(path, exif=exif)
    else:
        im.save(path)


def test_ingest_builds_a_training_folder_from_weighed_meals(tmp_path, taxonomy):
    tax_path, tax = taxonomy
    table = table_file(tmp_path, [["pap", "100", "1", "20", "3", "label", ""], ["boerewors", "300", "25", "2", "15", "label", ""]])
    photos = tmp_path / "phone"
    photos.mkdir()
    make_photo(photos / "a.jpg", orientation=6)  # stored sideways with a 'rotate 90' tag
    make_photo(photos / "b.jpg")
    make_photo(photos / "c.jpg")
    make_photo(photos / "d.jpg")
    meals = write_csv(tmp_path / "meals.csv", ["meal_id", "photos", "components", "calories", "fat", "carb", "protein"], [
        ["m1", "a.jpg|b.jpg", "pap:200|boerewors:100", "", "", "", ""],      # nutrients from the sheet
        ["m2", "c.jpg", "pap:200|chakalaka:50", "", "", "", ""],             # mass only: chakalaka has no row
        ["m3", "d.jpg", "boerewors:100", "999", "", "", ""],                 # calories measured, the rest from the sheet
    ])  # fmt: skip
    out = tmp_path / "sa"
    stats = ingest(meals, photos, out, tax_path, table, test_fraction=0.5, table_weight=0.4)
    assert stats["meals"] == 3 and stats["photos"] == 4

    with open(out / "index.csv", newline="") as f:
        rows = {r["path"]: r for r in csv.DictReader(f)}
    m1a, m1b, m2, m3 = rows["img/m1_1.jpg"], rows["img/m1_2.jpg"], rows["img/m2_1.jpg"], rows["img/m3_1.jpg"]
    assert (m1a["mass"], m1a["calories"], m1a["fat"], m1a["weight"]) == ("300.0", "500.0", "27.0", "0.4")
    assert m1a["split"] == m1b["split"] and m1a["dish_id"] == "m1"  # a meal's photos never straddle train and test
    assert (m2["mass"], m2["calories"], m2["weight"]) == ("250.0", "", "1")  # mass only, fully trusted
    assert (m3["calories"], m3["fat"], m3["weight"]) == ("999.0", "25.0", "0.4")  # measured value wins, others derived
    assert m1a["components"] == "pap:200|boerewors:100"

    with Image.open(out / "img" / "m1_1.jpg") as im:
        assert im.size == (40, 80)  # turned upright
        assert not im.getexif() and "exif" not in im.info  # camera metadata is gone

    def targets(split):
        try:
            return ExtraNutritionDataset(out, split, img_size=IMG).targets()
        except RuntimeError:  # the 50/50 split may leave one side empty
            return torch.zeros(0, TARGET_WIDTH)

    all_targets = torch.cat([targets("train"), targets("test")])
    assert len(all_targets) == 4
    # known labels: m1 has all five on each of its 2 photos, m2 only mass, m3 mass plus the four nutrients
    assert all_targets[:, KNOWN_START : KNOWN_START + N_OUT].sum().item() == 10 + 1 + 5


def test_ingest_rejects_bad_meals(tmp_path, taxonomy):
    tax_path, _ = taxonomy
    table = table_file(tmp_path, [])
    photos = tmp_path / "phone"
    photos.mkdir()
    make_photo(photos / "a.jpg")

    def run(rows, header=("meal_id", "photos", "components")):
        return ingest(write_csv(tmp_path / "m.csv", header, rows), photos, tmp_path / "out", tax_path, table)

    with pytest.raises(ValueError, match="did you mean"):
        run([["m1", "a.jpg", "pop:200"]])
    with pytest.raises(ValueError, match="not found"):
        run([["m1", "missing.jpg", "pap:200"]])
    with pytest.raises(ValueError, match="no photos"):
        run([["m1", "", "pap:200"]])
    with pytest.raises(ValueError, match="twice"):
        run([["m1", "a.jpg", "pap:200"], ["m1", "a.jpg", "pap:100"]])
    with pytest.raises(ValueError, match="non-negative"):
        run([["m1", "a.jpg", "pap:200", "-1"]], header=("meal_id", "photos", "components", "calories"))


def test_is_test_is_deterministic_and_roughly_proportional():
    keys = [f"meal{i}" for i in range(2000)]
    assert [is_test(k, 0.3) for k in keys] == [is_test(k, 0.3) for k in keys]
    assert 0.25 < sum(is_test(k, 0.3) for k in keys) / len(keys) < 0.35
    assert not any(is_test(k, 0.0) for k in keys) and all(is_test(k, 1.0) for k in keys)


def test_load_rgb_uint8_respects_the_exif_rotation_tag(tmp_path):
    # a left half red / right half blue picture stored with a 'rotate 90 clockwise' tag looks top-red / bottom-blue once upright
    im = Image.new("RGB", (40, 20))
    im.paste((255, 0, 0), (0, 0, 20, 20))
    im.paste((0, 0, 255), (20, 0, 40, 20))
    exif = Image.Exif()
    exif[0x0112] = 6
    im.save(tmp_path / "q.jpg", exif=exif, quality=100)
    with Image.open(tmp_path / "q.jpg") as raw:
        assert raw.size == (40, 20)  # the file really is stored sideways
    arr = load_rgb_uint8(tmp_path / "q.jpg", 32)
    assert arr[4, 16, 0] > 200 and arr[28, 16, 2] > 200


# --- Commons downloader --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name,expected", [
    ("CC BY 4.0", "cc-by"), ("CC BY 2.0", "cc-by"), ("cc-by-3.0", "cc-by"), ("CC BY-SA 4.0", "cc-by-sa"),
    ("CC BY-SA 3.0", "cc-by-sa"), ("CC0", "pd"), ("Public domain", "pd"), ("PD", "pd"),
    ("CC BY-NC 2.0", None), ("CC BY-ND 2.0", None), ("CC BY-NC-SA 4.0", None), ("GFDL", None), ("", None), ("All rights reserved", None),
])  # fmt: skip
def test_classify_licence(name, expected):
    assert pc.classify_licence(name) == expected


def commons_page(page_id, rank, licence, author="<a href='x'>Thandi &amp; Co</a>", sha="a", mime="image/jpeg", size=(800, 600)):
    return str(page_id), {
        "pageid": page_id, "index": rank, "title": f"File:{page_id}.jpg",
        "imageinfo": [{
            "url": f"https://upload.example/{page_id}.jpg", "thumburl": f"https://thumb.example/{page_id}.jpg",
            "descriptionurl": f"https://commons.example/File:{page_id}.jpg", "mime": mime, "width": size[0], "height": size[1], "sha1": sha,
            "extmetadata": {"LicenseShortName": {"value": licence}, "Artist": {"value": author}, "LicenseUrl": {"value": "https://cc.example"}},
        }],
    }  # fmt: skip


def fake_commons(tmp_path, results):
    """fetch / fetch_file stand-ins serving ``results[term]`` (a list of page tuples) and recording calls."""
    calls = {"search": [], "files": []}

    def fetch(params):
        term = params["gsrsearch"].removesuffix(" filetype:bitmap")
        calls["search"].append(term)
        return {"query": {"pages": dict(results.get(term, []))}}

    def fetch_file(url, dest):
        calls["files"].append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (1, 2, 3)).save(buf, format="JPEG")
        dest.write_bytes(buf.getvalue())

    return fetch, fetch_file, calls


def test_parse_candidates_cleans_html_and_orders_by_rank():
    payload = {"query": {"pages": dict([commons_page(2, 5, "CC BY 4.0"), commons_page(1, 1, "CC BY-SA 4.0")])}}
    first, second = pc.parse_candidates(payload)
    assert (first.page_id, second.page_id) == (1, 2)
    assert second.author == "Thandi & Co" and second.licence_class == "cc-by" and first.licence_class == "cc-by-sa"
    assert second.file_url == "https://thumb.example/2.jpg"


def test_download_keeps_only_allowed_licences_dedupes_and_records_credits(tmp_path, taxonomy):
    tax_path, _ = taxonomy
    results = {
        "pap maize porridge": [
            commons_page(1, 1, "CC BY 4.0", sha="s1"),
            commons_page(2, 2, "CC BY-NC 2.0", sha="s2"),       # NonCommercial: never
            commons_page(3, 3, "CC BY-SA 4.0", sha="s3"),       # only with --allow-sharealike
            commons_page(4, 4, "CC BY 4.0", sha="s1"),          # same file as 1 re-uploaded
            commons_page(5, 5, "CC BY 4.0", sha="s5", size=(100, 100)),  # too small
            commons_page(6, 6, "CC BY 4.0", sha="s6", author=""),        # no credit to give
            commons_page(7, 7, "CC0", sha="s7", author=""),              # public domain needs none
            commons_page(8, 8, "CC BY 4.0", sha="s8", mime="image/svg+xml"),
        ],
        "stywe pap": [commons_page(9, 1, "CC BY 4.0", sha="s9")],
        "boerewors": [commons_page(10, 1, "CC BY 4.0", sha="s7")],  # already downloaded under pap: skipped
    }  # fmt: skip
    fetch, fetch_file, calls = fake_commons(tmp_path, results)
    out = tmp_path / "commons"
    counts = pc.download(out, tax_path, fetch, fetch_file, per_dish=3)
    assert counts == {"pap": 3, "boerewors": 0, "chakalaka": 0}
    assert sorted(p.name for p in (out / "img" / "pap").iterdir()) == ["1.jpg", "7.jpg", "9.jpg"]
    with open(out / "attribution.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["path"] for r in rows] == ["img/pap/1.jpg", "img/pap/7.jpg", "img/pap/9.jpg"]
    assert rows[0]["author"] == "Thandi & Co" and rows[0]["licence"] == "CC BY 4.0" and rows[0]["page_url"].endswith("File:1.jpg")

    # share-alike is opt-in
    fetch, fetch_file, _ = fake_commons(tmp_path, results)
    counts = pc.download(tmp_path / "commons2", tax_path, fetch, fetch_file, per_dish=10, allow_sharealike=True)
    assert counts["pap"] == 4 and (tmp_path / "commons2" / "img" / "pap" / "3.jpg").exists()


def test_download_resumes_and_does_not_bring_back_photos_you_deleted(tmp_path, taxonomy):
    tax_path, _ = taxonomy
    results = {"pap maize porridge": [commons_page(i, i, "CC BY 4.0", sha=f"s{i}") for i in range(1, 6)]}
    out = tmp_path / "commons"
    fetch, fetch_file, calls = fake_commons(tmp_path, results)
    pc.download(out, tax_path, fetch, fetch_file, per_dish=3)
    (out / "img" / "pap" / "1.jpg").unlink()  # you looked at it and it was not pap
    fetch, fetch_file, calls = fake_commons(tmp_path, results)
    counts = pc.download(out, tax_path, fetch, fetch_file, per_dish=3)
    assert counts["pap"] == 3 and len(calls["files"]) == 1  # topped up with one new photo, not the rejected one
    assert not (out / "img" / "pap" / "1.jpg").exists() and (out / "img" / "pap" / "4.jpg").exists()


def test_index_lists_surviving_photos_and_splits_by_author(tmp_path, taxonomy):
    tax_path, _ = taxonomy
    pages = [commons_page(i, i, "CC BY 4.0", author=f"author{i % 20}", sha=f"s{i}") for i in range(1, 41)]
    fetch, fetch_file, _ = fake_commons(tmp_path, {"pap maize porridge": pages})
    out = tmp_path / "commons"
    pc.download(out, tax_path, fetch, fetch_file, per_dish=40)
    (out / "img" / "pap" / "5.jpg").unlink()
    assert pc.build_index(out, test_fraction=0.3) == {"pap": 39}
    with open(out / "index.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    authors: dict[str, set[str]] = {}
    with open(out / "attribution.csv", newline="") as f:
        by_path = {r["path"]: r["author"] for r in csv.DictReader(f)}
    for r in rows:
        authors.setdefault(by_path[r["path"]], set()).add(r["split"])
    assert all(len(s) == 1 for s in authors.values())  # one author is never on both sides
    assert {"test", ""} == {r["split"] for r in rows} and all(r["source"] == "commons" for r in rows)
    with pytest.raises(FileNotFoundError):
        pc.build_index(tmp_path / "nowhere")


def test_commons_photos_with_no_labels_train_only_the_food_output_and_report_food_recall(tmp_path):
    _fake_cache(tmp_path / "n5k")
    _fake_openimages(tmp_path / "oi")
    sa = tmp_path / "sa_commons"
    (sa / "img").mkdir(parents=True)
    rows = []
    for i in range(6):
        Image.new("RGB", (120, 90), (30 * i, 90, 30)).save(sa / "img" / f"{i}.jpg")
        rows.append([f"img/{i}.jpg", "test" if i >= 4 else "", "pap", "commons"])
    write_csv(sa / "index.csv", ["path", "split", "dish_id", "source"], rows)
    ds = ExtraNutritionDataset(sa, "train", img_size=IMG)
    assert ds.targets()[:, KNOWN_START : KNOWN_START + N_OUT].sum() == 0 and (ds.targets()[:, N_OUT] == 1).all()

    out = tmp_path / "run"
    train_main(
        ["--data-root", str(tmp_path / "n5k"), "--openimages-root", str(tmp_path / "oi"), "--extra-data", str(sa), "--no-pretrained",
         "--epochs", "1", "--batch-size", "8", "--img-size", str(IMG), "--load-threads", "1", "--device", "cpu",
         "--val-fraction", "0.2", "--out", str(out)]
    )  # fmt: skip
    report = json.loads((out / "test_metrics.json").read_text())["extra_sa_commons"]
    assert report["images"] == 2 and "food_recall" in report and "mae_calories" not in report


def test_holdout_data_is_scored_with_a_median_guess_and_never_trained_on(tmp_path, capsys):
    import re

    _fake_cache(tmp_path / "n5k")
    sa = tmp_path / "sa_meals"
    (sa / "img").mkdir(parents=True)
    rows = []
    for i in range(5):
        Image.new("RGB", (120, 90), (30 * i, 90, 30)).save(sa / "img" / f"{i}.jpg")
        rows.append([f"img/{i}.jpg", "250", "480", "test" if i >= 3 else ""])
    write_csv(sa / "index.csv", ["path", "mass", "calories", "split"], rows)

    common = ["--data-root", str(tmp_path / "n5k"), "--no-pretrained", "--epochs", "1", "--batch-size", "8", "--img-size", str(IMG),
              "--load-threads", "1", "--device", "cpu", "--val-fraction", "0.2"]  # fmt: skip

    def n_train():
        return int(re.search(r"images: (\d+) train", capsys.readouterr().out).group(1))

    train_main([*common, "--out", str(tmp_path / "plain")])
    plain = n_train()
    out = tmp_path / "run"
    train_main([*common, "--holdout-data", str(sa), "--out", str(out)])
    assert n_train() == plain  # the SA folder added nothing to training, not even its non-test rows

    report = json.loads((out / "test_metrics.json").read_text())
    assert report["extra_sa_meals"]["images"] == 2 and "mae_calories" in report["extra_sa_meals"]
    guess = report["extra_sa_meals_guess"]  # the Nutrition5k median (250 g, about 300 kcal) against 250 g / 480 kcal meals
    assert guess["mae_mass"] == pytest.approx(0.0) and 170 < guess["mae_calories"] < 185

    with pytest.raises(SystemExit):
        train_main([*common, "--extra-data", str(sa), "--holdout-data", str(sa), "--out", str(tmp_path / "both")])
