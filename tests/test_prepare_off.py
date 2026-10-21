import csv
import gzip
import json

from calorie_model import prepare_off as po


def product(code="6001", name="Orange Juice", **over):
    base = {
        "code": code, "product_name": name, "brands": "Ceres, Other", "countries_tags": ["en:south-africa", "en:namibia"],
        "categories_tags": ["en:beverages", "en:fruit-juices"], "serving_quantity": "250", "serving_size": "250 ml",
        "image_front_url": f"https://images.example/{code}.jpg",
        "nutriments": {"energy-kcal_100g": 45, "proteins_100g": 0.5, "carbohydrates_100g": 10.5, "fat_100g": 0.1, "sugars_100g": 10, "sodium_100g": 0.01},
    }  # fmt: skip
    base.update(over)
    return base


def test_keeps_south_african_products_with_their_nutrition():
    row = po.parse_product(product())
    assert row["barcode"] == "6001" and row["brand"] == "Ceres" and row["group"] == "juice"
    assert (row["calories_100g"], row["protein_100g"], row["carb_100g"], row["fat_100g"]) == (45, 0.5, 10.5, 0.1)
    assert row["sodium_mg_100g"] == 10 and row["serving_g"] == 250
    assert row["page_url"].endswith("/product/6001") and "CC BY-SA" in row["licence"]


def test_rejects_other_countries_and_unusable_energy():
    assert po.parse_product(product(countries_tags=["en:france"])) is None
    assert po.parse_product(product(nutriments={})) is None
    assert po.parse_product(product(nutriments={"energy-kcal_100g": 4000})) is None
    assert po.parse_product(product(product_name="  ")) is None
    assert po.parse_product(product(code="")) is None


def test_uses_kilojoules_when_kcal_is_missing():
    row = po.parse_product(product(nutriments={"energy_100g": 2092, "proteins_100g": "6"}))
    assert row["calories_100g"] == 500 and row["protein_100g"] == 6


def test_groups_sweets_juice_chocolate_and_falls_back_to_other():
    g = lambda *tags: po.group_of(tags)  # noqa: E731
    assert g("en:chocolates") == "chocolate"
    assert g("en:candies", "en:snacks") == "sweets"
    assert g("en:fruit-juices") == "juice"
    assert g("en:carbonated-drinks") == "soft_drink"
    assert g("en:biltong") == "snacks"
    assert g("en:mystery") == "other"


def test_streams_only_south_african_lines_from_a_gzip_export(tmp_path):
    lines = [product("1"), product("2", countries_tags=["en:france"]), product("3", categories_tags=["en:chocolates"]), {"code": "4", "countries_tags": ["en:south-africa"]}]
    dump = tmp_path / "off.jsonl.gz"
    with gzip.open(dump, "wt") as f:
        f.write("\n".join(json.dumps(x) for x in lines) + "\nnot json en:south-africa\n")
    rows = list(po.iter_south_african(po.open_source(str(dump), None)))
    assert [r["barcode"] for r in rows] == ["1", "3"]  # the French one, the one with no energy and the junk line are skipped


def test_writes_a_csv_and_downloads_photos_politely(tmp_path):
    rows = [po.parse_product(product("1")), po.parse_product(product("2", image_front_url="")), po.parse_product(product("3"))]
    seen = []
    got = po.download_images(rows, tmp_path, "me@example.com", limit=1, delay=0, fetch=lambda url: seen.append(url) or b"\xff\xd8\xff")
    assert got == 1 and seen == ["https://images.example/1.jpg"]
    assert (tmp_path / "img" / "juice" / "1.jpg").exists() and rows[0]["image_path"] == "img/juice/1.jpg" and rows[2]["image_path"] == ""
    po.write_csv(rows, tmp_path / "products.csv")
    back = list(csv.DictReader(open(tmp_path / "products.csv", encoding="utf-8")))
    assert [r["barcode"] for r in back] == ["1", "2", "3"] and list(back[0]) == po.COLUMNS
    assert "juice 3" in po.summarize(rows)
