import numpy as np
import pytest
import torch
from PIL import Image

from calorie_model import OUTPUT_NAMES, TARGETS, NutritionNet
from calorie_model.data import TARGET_WIDTH, Nutrition5kDataset, OpenImagesDataset, SyntheticDataset, load_rgb_uint8
from calorie_model.model import count_parameters
from calorie_model.train import compute_loss, main as train_main

IMG = 96  # small so tests stay fast


@pytest.fixture(scope="module")
def net():
    """A model without the food output: the original five-output layout."""
    return NutritionNet(pretrained=False, food_head=False).eval()


def test_output_shape_and_nonnegative(net):
    out = net(torch.rand(3, 3, IMG, IMG))
    assert out.shape == (3, len(TARGETS))
    assert (out >= 0).all()


def test_food_head_adds_a_sixth_logit_and_keeps_nutrition_nonnegative():
    out = NutritionNet(pretrained=False).eval()(torch.rand(3, 3, IMG, IMG))
    assert out.shape == (3, len(OUTPUT_NAMES)) and OUTPUT_NAMES[-1] == "food"
    assert (out[:, : len(TARGETS)] >= 0).all()  # the last column is a free-signed logit


def test_stays_lightweight():
    # Phone budget guard: the default model must stay well under 5M parameters.
    assert count_parameters(NutritionNet(pretrained=False)) < 5_000_000


def test_unknown_backbone_rejected():
    with pytest.raises(ValueError):
        NutritionNet("resnet9000", pretrained=False)


def test_synthetic_training_runs_and_checkpoints(tmp_path):
    train_main(
        ["--synthetic", "--no-pretrained", "--epochs", "2", "--batch-size", "16",
         "--img-size", str(IMG), "--load-threads", "1", "--device", "cpu", "--out", str(tmp_path)]
    )
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "history.json").exists()
    import json

    report = json.loads((tmp_path / "test_metrics.json").read_text())
    assert {"all", "baseline_median"} <= set(report) and "mae_calories" in report["all"]


def test_training_reduces_loss():
    torch.manual_seed(0)
    model = NutritionNet(pretrained=False)
    ds = SyntheticDataset(n=32, img_size=IMG)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.stack([ds[i][1] for i in range(len(ds))])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = last = None
    model.train()
    for _ in range(25):
        loss = compute_loss(model(x), y, torch.ones(len(TARGETS)), gate_weight=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    assert last < first * 0.7


def test_onnx_export_matches_torch(tmp_path):
    pytest.importorskip("onnxruntime")
    from calorie_model.export import check_parity, export_onnx

    net = NutritionNet(pretrained=False).eval()
    out = tmp_path / "m.onnx"
    export_onnx(net, IMG, out)
    assert check_parity(net, out, IMG) < 1e-2


def test_deploy_contract_is_uint8_nhwc_in_real_units(net):
    from calorie_model.export import DeployModel
    from calorie_model.model import scale_vector

    pixels = np.random.default_rng(0).integers(0, 256, (2, IMG, IMG, 3), dtype=np.uint8)
    with torch.no_grad():
        got = DeployModel(net).eval()(torch.from_numpy(pixels))
        chw = torch.from_numpy(pixels).permute(0, 3, 1, 2).float() / 255.0
        want = net(chw) * scale_vector()
    assert got.shape == (2, len(TARGETS))
    assert torch.allclose(got, want, atol=1e-4)


def test_predict_end_to_end(tmp_path):
    pytest.importorskip("onnxruntime")
    from calorie_model.export import export_onnx
    from calorie_model.predict import predict

    export_onnx(NutritionNet(pretrained=False).eval(), IMG, tmp_path / "m.onnx")
    Image.new("RGB", (160, 120), (200, 120, 60)).save(tmp_path / "food.png")
    result = predict(tmp_path / "m.onnx", tmp_path / "food.png")
    assert list(result) == list(OUTPUT_NAMES)  # five nutrition values and the food probability
    assert all(v >= 0 for v in result.values()) and result["food"] <= 1


def test_load_rgb_uint8_shape_and_dtype(tmp_path):
    Image.new("RGB", (50, 80), (1, 2, 3)).save(tmp_path / "a.png")
    arr = load_rgb_uint8(tmp_path / "a.png", IMG)
    assert arr.shape == (IMG, IMG, 3) and arr.dtype == np.uint8


def _fake_cache(root):
    """A tiny prepared cache: 5 train dishes and 2 test dishes, each with an overhead and a side image."""
    import csv

    rows, labels = [], []
    for i in range(7):
        dish, split = f"dish_{i}", "test" if i >= 5 else "train"
        (root / "cache" / dish).mkdir(parents=True)
        labels.append((dish, 300.0 + i, 250.0, 10.0, 40.0, 12.0))
        for view, name in (("overhead", "overhead.jpg"), ("side", "side_A_0.jpg")):
            Image.new("RGB", (64, 48), (10, 20, 30)).save(root / "cache" / dish / name)
            rows.append((dish, split, view, f"cache/{dish}/{name}"))
    with open(root / "index.csv", "w", newline="") as f:
        csv.writer(f).writerows([("dish_id", "split", "view", "path"), *rows])
    with open(root / "labels.csv", "w", newline="") as f:
        csv.writer(f).writerows([("dish_id", *TARGETS), *labels])


def test_nutrition5k_splits_are_disjoint_by_dish(tmp_path):
    _fake_cache(tmp_path)
    sets = {s: Nutrition5kDataset(tmp_path, s, img_size=IMG, val_fraction=0.2) for s in ("train", "val", "test")}
    dishes = {s: {p.parent.name for p, _ in ds.samples} for s, ds in sets.items()}
    assert len(sets["val"]) == 2 and len(sets["train"]) == 8 and len(sets["test"]) == 4  # 1 val dish, 2 views each
    assert not dishes["train"] & dishes["val"] and not dishes["train"] & dishes["test"]
    assert dishes["test"] == {"dish_5", "dish_6"}  # only the official test dishes
    again = Nutrition5kDataset(tmp_path, "val", img_size=IMG, val_fraction=0.2)
    assert [p for p, _ in again.samples] == [p for p, _ in sets["val"].samples]  # deterministic


def test_nutrition5k_loader_labels_and_view_filter(tmp_path):
    _fake_cache(tmp_path)
    ds = Nutrition5kDataset(tmp_path, "test", img_size=IMG, views=("overhead",))
    assert len(ds) == 2 and all(p.name == "overhead.jpg" for p, _ in ds.samples)
    x, y = ds[0]
    assert x.shape == (3, IMG, IMG) and 0 <= x.min() and x.max() <= 1
    # Targets are scaled units: 305 kcal / 500 = 0.61, 250 g / 400 = 0.625.
    assert torch.allclose(y[:2], torch.tensor([0.61, 0.625]), atol=1e-6)
    assert ds.targets().shape == (2, TARGET_WIDTH)  # 5 nutrition values + food flag + has-nutrition flag
    assert ds.targets()[0, len(TARGETS):].tolist() == [1.0, 1.0]


def test_nutrition5k_loader_explains_missing_images(tmp_path):
    _fake_cache(tmp_path)
    with pytest.raises(RuntimeError, match="prepare_data"):
        Nutrition5kDataset(tmp_path, "test", img_size=IMG, views=("nonexistent",))


def _h264_stream(frames=12, size=(128, 96)):
    import io

    av = pytest.importorskip("av")
    buf = io.BytesIO()
    out = av.open(buf, "w", format="h264")
    stream = out.add_stream("libx264", rate=30)
    stream.width, stream.height, stream.pix_fmt = size[0], size[1], "yuv420p"
    for i in range(frames):
        arr = np.full((size[1], size[0], 3), (i * 20) % 255, dtype=np.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
            out.mux(packet)
    for packet in stream.encode():
        out.mux(packet)
    out.close()
    return buf.getvalue()


def test_decode_frames_drops_the_partial_last_frame_of_a_cut_stream():
    from calorie_model.prepare_data import decode_frames

    data = _h264_stream(frames=12)
    assert len(decode_frames(data, truncated=False, count=100)) == 12
    assert len(decode_frames(data, truncated=True, count=100)) == 11
    assert len(decode_frames(data, truncated=False, count=1)) == 1
    assert decode_frames(b"not video at all", truncated=False) == []


def test_prepare_data_end_to_end_against_a_fake_bucket(tmp_path, monkeypatch):
    import csv
    import io

    from calorie_model import prepare_data as pd_

    def png(color):
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), color).save(buf, format="PNG")
        return buf.getvalue()

    ingredient = ",ingr_1,rice,100,130,0.3,28,2.5"  # variable-length tail after the six dish columns
    bucket = {
        "metadata/dish_metadata_cafe1.csv": f"dish_a,300,200,10,30,15,1{ingredient}\ndish_b,200,150,5,25,10,0\ndish_c,400,300,20,40,20,0\n".encode(),
        "metadata/dish_metadata_cafe2.csv": b"dish_x,100,80,2,10,5,0\n",
        "dish_ids/splits/rgb_train_ids.txt": b"dish_a\ndish_b\ndish_unlabelled\n",
        "dish_ids/splits/rgb_test_ids.txt": b"dish_c\n",
        "imagery/realsense_overhead/dish_a/rgb.png": png((200, 10, 10)),
        "imagery/realsense_overhead/dish_c/rgb.png": png((10, 200, 10)),
        "imagery/side_angles/dish_a/camera_A.h264": _h264_stream(),
    }

    def fake_fetch(url, byte_limit=None, retries=4):
        data = bucket.get(url.removeprefix(pd_.BASE))
        return None if data is None else (data[:byte_limit] if byte_limit else data)

    monkeypatch.setattr(pd_, "fetch", fake_fetch)

    out = tmp_path / "n5k"
    pd_.prepare(out, {"overhead", "side"}, size=32, side_bytes=100_000, side_frames=1, workers=2)
    index = list(csv.DictReader(open(out / "index.csv")))
    got = {(r["dish_id"], r["split"], r["view"]) for r in index}
    assert got == {
        ("dish_a", "train", "overhead"),
        ("dish_a", "train", "side"),
        ("dish_c", "test", "overhead"),
    }  # dish_b has no media; dish_x has none either; unlabelled ids are ignored
    assert Image.open(out / "cache/dish_a/overhead.jpg").size == (32, 32)
    labels = {r["dish_id"]: r for r in csv.DictReader(open(out / "labels.csv"))}
    assert set(labels) == {"dish_a", "dish_b", "dish_c", "dish_x"}  # dish_x joins train as extra data
    assert float(labels["dish_a"]["calories"]) == 300.0 and float(labels["dish_a"]["protein"]) == 15.0

    pd_.prepare(tmp_path / "n5k_noextra", {"overhead"}, size=32, extra=False, workers=2)
    assert "dish_x" not in open(tmp_path / "n5k_noextra" / "labels.csv").read()


def test_int8_quantization_shrinks_model_and_still_runs(tmp_path):
    pytest.importorskip("onnxruntime")
    import onnx
    import onnxruntime as ort

    from calorie_model.export import export_onnx, quantize_int8, random_pixels, remove_identities

    export_onnx(NutritionNet(pretrained=False).eval(), IMG, tmp_path / "m.onnx")
    remove_identities(tmp_path / "m.onnx", tmp_path / "noid.onnx")
    assert not any(n.op_type == "Identity" for n in onnx.load(tmp_path / "noid.onnx").graph.node)

    quantize_int8(tmp_path / "m.onnx", tmp_path / "m.int8.onnx", IMG, None)
    assert (tmp_path / "m.int8.onnx").stat().st_size < 0.5 * (tmp_path / "m.onnx").stat().st_size
    assert not (tmp_path / "m.int8.prep.onnx").exists()  # temp file cleaned up
    out = ort.InferenceSession(str(tmp_path / "m.int8.onnx"), providers=["CPUExecutionProvider"]).run(
        None, {"image": random_pixels(1, IMG)}
    )[0]
    assert out.shape == (1, len(OUTPUT_NAMES)) and np.isfinite(out).all()


def test_tflite_export_matches_torch_and_takes_uint8(tmp_path):
    # Needs the optional converter (pip install litert-torch ai-edge-litert); skipped otherwise.
    pytest.importorskip("litert_torch")
    pytest.importorskip("ai_edge_litert")
    from ai_edge_litert.interpreter import Interpreter

    from calorie_model.export_tflite import check_parity, export_tflite

    net = NutritionNet(pretrained=False).eval()
    export_tflite(net, IMG, tmp_path / "m.tflite")
    assert check_parity(net, tmp_path / "m.tflite", IMG) < 1e-1

    interp = Interpreter(model_path=str(tmp_path / "m.tflite"))
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    assert inp["dtype"] == np.uint8 and tuple(inp["shape"]) == (1, IMG, IMG, 3)
    assert out["dtype"] == np.float32 and tuple(out["shape"]) == (1, len(OUTPUT_NAMES))


def test_gpu_augmentation_identity_and_flips_are_exact():
    from calorie_model.batches import augment, identity_params

    x = torch.rand(3, 3, 32, 32)
    p = identity_params(3)
    assert torch.allclose(augment(x, p), x, atol=1e-4)
    assert torch.allclose(augment(x, {**p, "flip_x": -torch.ones(3)}), x.flip(-1), atol=1e-4)
    assert torch.allclose(augment(x, {**p, "flip_y": -torch.ones(3)}), x.flip(-2), atol=1e-4)


def test_gpu_augmentation_quarter_turn_matches_rot90():
    import math

    from calorie_model.batches import augment, identity_params

    x = torch.rand(2, 3, 32, 32)
    out = augment(x, {**identity_params(2), "angle": torch.full((2,), math.pi / 2)})
    assert torch.allclose(out, x.rot90(1, (2, 3)), atol=1e-4) or torch.allclose(out, x.rot90(-1, (2, 3)), atol=1e-4)


def test_gpu_augmentation_zoom_crops_the_centre():
    from calorie_model.batches import augment, identity_params

    x = torch.zeros(1, 3, 32, 32)
    x[:, :, 12:20, 12:20] = 1.0  # bright centre square
    zoomed = augment(x, {**identity_params(1), "w": torch.tensor([0.5]), "h": torch.tensor([0.5])})
    assert zoomed.mean() > x.mean() * 3  # the centre now fills far more of the frame


def test_gpu_augmentation_colour_ops_behave():
    from calorie_model.batches import augment, identity_params

    x = torch.rand(2, 3, 16, 16)
    p = identity_params(2)
    assert augment(x, {**p, "brightness": torch.full((2,), 0.5)}).mean() < x.mean()
    grey = augment(x, {**p, "saturation": torch.zeros(2)})
    assert torch.allclose(grey[:, 0], grey[:, 1], atol=1e-4) and torch.allclose(grey[:, 1], grey[:, 2], atol=1e-4)
    assert augment(x, {**p, "hue": torch.full((2,), 0.3)}).std() > 0  # runs, stays finite
    assert torch.isfinite(augment(x, {**p, "hue": torch.full((2,), 0.3)})).all()


def test_sampled_augmentation_keeps_shape_and_range():
    from calorie_model.batches import augment, sample_params

    torch.manual_seed(0)
    x = torch.rand(16, 3, 40, 40)
    out = augment(x, sample_params(16, "cpu"))
    assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1
    assert not torch.allclose(out, x)


def test_batches_cover_every_sample_once_and_respect_drop_last():
    from calorie_model.batches import Batches

    images = torch.arange(10, dtype=torch.uint8).view(10, 1, 1, 1).expand(10, 3, 4, 4).contiguous()
    targets = torch.arange(10, dtype=torch.float32).view(10, 1).expand(10, 5).contiguous()

    plain = Batches(images, targets, batch_size=4)
    assert len(plain) == 3 and [len(y) for _, y in plain] == [4, 4, 2]
    assert torch.equal(torch.cat([y[:, 0] for _, y in plain]), torch.arange(10, dtype=torch.float32))

    shuffled = Batches(images, targets, batch_size=4, shuffle=True)
    seen = torch.cat([y[:, 0] for _, y in shuffled])
    assert len(shuffled) == 2 and len(seen) == 8 and len(set(seen.tolist())) == 8  # last partial batch dropped

    x, y = next(iter(plain))
    assert x.dtype == torch.float32 and x.max() <= 1 and torch.allclose(x[:, 0, 0, 0] * 255, y[:, 0])  # image i matches target i


def test_load_bank_matches_dataset_images_exactly(tmp_path):
    from calorie_model.batches import load_bank

    _fake_cache(tmp_path)
    ds = Nutrition5kDataset(tmp_path, "train", img_size=IMG, val_fraction=0.2)
    images, targets = load_bank(ds, "cpu", threads=2)
    assert images.dtype == torch.uint8 and images.shape == (len(ds), 3, IMG, IMG)
    assert torch.equal(images[3].float() / 255, ds[3][0]) and torch.equal(targets, ds.targets())


def test_bench_runs_on_cpu(capsys):
    from calorie_model.bench import main as bench_main

    bench_main(["--device", "cpu", "--batch-size", "4", "--img-size", "64", "--steps", "1"])
    out = capsys.readouterr().out
    assert "augmentation alone" in out and "img/s" in out


def test_train_fraction_keeps_fewer_dishes_and_never_touches_val_or_test(tmp_path):
    import csv

    _fake_cache(tmp_path)
    # extend the fake cache to 20 train dishes so a fraction is meaningful
    rows = list(csv.reader(open(tmp_path / "index.csv")))
    labels = list(csv.reader(open(tmp_path / "labels.csv")))
    for i in range(10, 25):
        dish = f"dish_{i}"
        (tmp_path / "cache" / dish).mkdir()
        Image.new("RGB", (64, 48)).save(tmp_path / "cache" / dish / "overhead.jpg")
        rows.append([dish, "train", "overhead", f"cache/{dish}/overhead.jpg"])
        labels.append([dish, "300", "250", "10", "40", "12"])
    csv.writer(open(tmp_path / "index.csv", "w", newline="")).writerows(rows)
    csv.writer(open(tmp_path / "labels.csv", "w", newline="")).writerows(labels)

    kw = dict(img_size=IMG, views=("overhead",), val_fraction=0.2)
    full = Nutrition5kDataset(tmp_path, "train", **kw)
    half = Nutrition5kDataset(tmp_path, "train", train_fraction=0.5, **kw)
    dishes = lambda ds: {p.parent.name for p, _ in ds.samples}  # noqa: E731
    assert len(half) == round(len(full) * 0.5) and dishes(half) < dishes(full)
    assert dishes(half) == dishes(Nutrition5kDataset(tmp_path, "train", train_fraction=0.5, **kw))  # deterministic
    for split in ("val", "test"):  # unaffected
        assert len(Nutrition5kDataset(tmp_path, split, train_fraction=0.1, **kw)) == len(Nutrition5kDataset(tmp_path, split, **kw))


# ---------------------------------------------------------------------------------------------
# Food / not-food output
# ---------------------------------------------------------------------------------------------


def _fake_openimages(root):
    """A tiny Open Images cache: 'train' pool of food and non-food photos, and a held-out 'test' set."""
    import csv

    rows = []
    layout = {
        "train": {"food": 8, "animal": 3, "person": 3, "other": 4},
        "test": {"food": 3, "animal": 2, "person": 2, "other": 2},
    }
    for split, groups in layout.items():
        for group, n in groups.items():
            for i in range(n):
                image_id = f"{split}_{group}_{i}"
                (root / "cache" / split).mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (64, 48), (50 + 30 * i, 100, 150)).save(root / "cache" / split / f"{image_id}.jpg")
                label = "food" if group == "food" else "nonfood"
                rows.append((image_id, split, label, group, f"cache/{split}/{image_id}.jpg", split, "by/2.0", "Someone", "https://x", "t"))
    with open(root / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("image_id", "split", "label", "group", "path", "subset", "license", "author", "landing_url", "title"))
        w.writerows(rows)


def test_openimages_dataset_labels_splits_and_groups(tmp_path):
    _fake_openimages(tmp_path)
    kw = dict(img_size=IMG, val_fraction=0.2)
    train, val, test = (OpenImagesDataset(tmp_path, s, **kw) for s in ("train", "val", "test"))
    assert len(train) + len(val) == 18 and len(test) == 9
    ids = lambda ds: {p.stem for p, _ in ds.samples}  # noqa: E731
    assert not ids(train) & ids(val) and all(i.startswith("test_") for i in ids(test))
    food = test.targets()[:, len(TARGETS)]  # the food column
    assert [g == "food" for g in test.groups] == [bool(v) for v in food]
    assert (test.targets()[:, len(TARGETS) + 1] == 0).all()  # no nutrition labels
    assert set(OpenImagesDataset(tmp_path, "test", img_size=IMG, groups=("animal",)).groups) == {"animal"}


def test_roc_auc_and_gate_metrics():
    from calorie_model.train import gate_metrics, roc_auc

    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert roc_auc(torch.tensor([0.9, 0.8, 0.2, 0.1]), labels) == 1.0
    assert roc_auc(torch.tensor([0.1, 0.2, 0.8, 0.9]), labels) == 0.0
    assert roc_auc(torch.tensor([0.5, 0.5, 0.5]), torch.ones(3)) != roc_auc(torch.tensor([0.5, 0.5, 0.5]), torch.ones(3))  # nan
    m = gate_metrics(torch.tensor([0.9, 0.4, 0.2, 0.7]), labels)
    assert m["food_recall"] == 0.5 and m["nonfood_rejected"] == 0.5 and m["gate_acc"] == 0.5
    assert gate_metrics(torch.tensor([0.9, 0.4, 0.2, 0.7]), labels, threshold=0.3)["food_recall"] == 1.0


def test_loss_ignores_nutrition_for_photos_without_labels_but_still_trains_the_food_output():
    scale = torch.ones(len(TARGETS))
    pred = torch.randn(4, 6)
    y = torch.zeros(4, 7)
    y[:2, :5] = torch.rand(2, 5)
    y[:2, 5] = 1.0  # two labelled food photos
    y[:2, 6] = 1.0  # ... with nutrition labels
    y[2:, 5] = torch.tensor([1.0, 0.0])  # a food photo and a non-food photo, no nutrition labels
    garbage = y.clone()
    garbage[2:, :5] = 999.0  # nutrition targets of unlabelled photos must not matter
    assert torch.allclose(compute_loss(pred, y, scale, 1.0), compute_loss(pred, garbage, scale, 1.0))
    only_labelled = compute_loss(pred[:2], y[:2], scale, 0.0)
    nutrition_part = compute_loss(pred, y, scale, 0.0)
    assert torch.allclose(nutrition_part, only_labelled)  # averaged over labelled photos only
    assert compute_loss(pred, y, scale, 1.0) > nutrition_part  # the food loss adds to it
    flipped = y.clone()
    flipped[2:, 5] = 1 - flipped[2:, 5]
    assert not torch.allclose(compute_loss(pred, y, scale, 1.0), compute_loss(pred, flipped, scale, 1.0))
    # a model without the food output ignores the food columns
    assert compute_loss(pred[:, :5], y, scale, 1.0).ndim == 0


def test_load_bank_concatenates_datasets_in_order(tmp_path):
    from calorie_model.batches import load_bank

    _fake_cache(tmp_path / "n5k")
    _fake_openimages(tmp_path / "oi")
    a = Nutrition5kDataset(tmp_path / "n5k", "test", img_size=IMG)
    b = OpenImagesDataset(tmp_path / "oi", "test", img_size=IMG)
    images, targets = load_bank([a, b], "cpu", threads=2)
    assert images.shape == (len(a) + len(b), 3, IMG, IMG) and images.dtype == torch.uint8
    assert torch.equal(targets, torch.cat([a.targets(), b.targets()]))
    assert torch.equal(images[len(a)].float() / 255, b[0][0])  # first Open Images photo sits right after the dishes


def _fake_openimages_server(monkeypatch, tmp_path):
    """Patch prepare_openimages.fetch to serve label files, image metadata and JPEGs from memory."""
    import io

    from calorie_model import prepare_openimages as po

    def jpeg(c):
        buf = io.BytesIO()
        Image.new("RGB", (90, 60), c).save(buf, format="JPEG")
        return buf.getvalue()

    def labels(rows):
        return ("ImageID,Source,LabelName,Confidence\n" + "\n".join(",".join(r) for r in rows) + "\n").encode()

    def meta(ids):
        head = "ImageID,Subset,OriginalURL,OriginalLandingURL,License,AuthorProfileURL,Author,Title,OriginalSize,OriginalMD5,Thumbnail300KURL,Rotation\n"
        return (head + "".join(f"{i},x,u,https://flickr/{i},https://creativecommons.org/licenses/by/2.0/,p,Author {i},Title {i},1,m,t,\n" for i in ids)).encode()

    FOOD, DOG, PERSON, OTHER, PLATE = "/m/02wbm", "/m/0bt9lr", "/m/01g317", "/m/0cmf2", "/m/050gv4"
    spec = {  # image -> (positive labels, extra rows)
        "a1": [FOOD], "a2": [FOOD, PLATE], "a3": [DOG], "a4": [PERSON], "a5": [OTHER],
        "a6": [PLATE],  # tableware but no food label: ambiguous, must be left out
        "a7": [],  # only a negative verification: never positively labelled, must be left out
    }
    rows = []
    for image_id, labs in spec.items():
        rows += [(image_id, "verification", m, "1") for m in labs]
        if not labs:
            rows.append((image_id, "verification", OTHER, "0"))
    bucket = {
        po.META_BASE + po.LABEL_FILES["test"]: labels(rows),
        po.META_BASE + po.IMAGE_FILES["test"]: meta(spec),
        po.META_BASE + po.LABEL_FILES["validation"]: labels([("v1", "verification", FOOD, "1"), ("v2", "verification", DOG, "1")]),
        po.META_BASE + po.IMAGE_FILES["validation"]: meta(["v1", "v2"]),
    }
    for subset, ids in (("test", list(spec)), ("validation", ["v1", "v2"])):
        for i in ids:
            bucket[po.IMAGE_URL.format(subset=subset, image_id=i)] = jpeg((10, 120, 200))
    calls = []

    def fake_fetch(url, byte_limit=None, retries=4):
        calls.append(url)
        return bucket.get(url)

    monkeypatch.setattr(po, "fetch", fake_fetch)
    return po, calls


def test_prepare_openimages_groups_excludes_ambiguous_and_records_attribution(tmp_path, monkeypatch):
    import csv

    po, calls = _fake_openimages_server(monkeypatch, tmp_path)
    assert po.classify({"x": {"/m/02wbm", "/m/050gv4"}, "y": {"/m/050gv4"}, "z": set(), "d": {"/m/0bt9lr"}}) == {"x": "food", "d": "animal"}

    out = tmp_path / "oi"
    po.prepare(out, scale=0.001, size=48, workers=2)  # scale tiny: one image per group
    rows = list(csv.DictReader(open(out / "index.csv")))
    by_id = {r["image_id"]: r for r in rows}
    assert "a6" not in by_id and "a7" not in by_id  # ambiguous and never-labelled images are excluded
    assert by_id["a3"]["group"] == "animal" and by_id["a3"]["label"] == "nonfood" and by_id["a3"]["split"] == "train"
    assert by_id["v1"]["label"] == "food" and by_id["v1"]["split"] == "test"  # validation subset is the held-out split
    assert by_id["a1"]["author"] == "Author a1" and "creativecommons" in by_id["a1"]["license"]
    assert Image.open(out / "cache/test/a1.jpg").size == (48, 48)

    before = len(calls)
    po.prepare(out, scale=0.001, size=48, workers=2)  # resumable: images already cached are not fetched again
    assert not any(u.endswith(".jpg") for u in calls[before:])


def test_training_with_the_food_output_end_to_end(tmp_path):
    import json

    _fake_cache(tmp_path / "n5k")
    _fake_openimages(tmp_path / "oi")
    out = tmp_path / "run"
    train_main(
        ["--data-root", str(tmp_path / "n5k"), "--openimages-root", str(tmp_path / "oi"), "--no-pretrained",
         "--epochs", "2", "--batch-size", "8", "--img-size", str(IMG), "--load-threads", "1", "--device", "cpu",
         "--val-fraction", "0.2", "--out", str(out)]
    )  # fmt: skip
    ckpt = torch.load(out / "best.pt", weights_only=True)
    assert ckpt["food_head"] is True
    report = json.loads((out / "test_metrics.json").read_text())
    gate = report["gate"]
    assert gate["images"] == 9 and set(gate["groups"]) == {"food", "animal", "person", "other"}
    assert "nutrition5k_test_recognised_as_food" in gate and set(gate["sweep"]) == {"0.20", "0.35", "0.50", "0.65", "0.80"}
    assert "mae_calories" in report["all"] and "baseline_median" in report  # nutrition outputs still reported


def test_food_head_deploy_output_and_legacy_checkpoints(tmp_path):
    from calorie_model.export import DeployModel, load_checkpoint, random_pixels

    px = torch.from_numpy(random_pixels(2, IMG))
    with torch.no_grad():
        out = DeployModel(NutritionNet(pretrained=False).eval())(px)
    assert out.shape == (2, 6) and ((out[:, 5] >= 0) & (out[:, 5] <= 1)).all()  # a probability

    legacy = NutritionNet(pretrained=False, food_head=False)  # a checkpoint from before the food output existed
    torch.save({"state_dict": legacy.state_dict(), "backbone": "mobilenet_v3_small", "img_size": IMG}, tmp_path / "old.pt")
    net, size = load_checkpoint(tmp_path / "old.pt")
    assert net.food_head is False and size == IMG
    with torch.no_grad():
        assert DeployModel(net)(px).shape == (2, 5)


def test_check_photos_reports_not_food_below_the_threshold(tmp_path, capsys):
    from calorie_model.check_photos import describe, expand, main as check_main

    net = NutritionNet(pretrained=False)
    torch.save({"state_dict": net.state_dict(), "backbone": "mobilenet_v3_small", "img_size": IMG, "food_head": True}, tmp_path / "m.pt")
    (tmp_path / "photos").mkdir()
    for name in ("dog.jpg", "lunch.png"):
        Image.new("RGB", (80, 60), (120, 90, 60)).save(tmp_path / "photos" / name)
    (tmp_path / "photos" / "notes.txt").write_text("not an image")

    assert [p.name for p in expand([str(tmp_path / "photos")])] == ["dog.jpg", "lunch.png"]  # folder -> images only
    assert [p.name for p in expand([str(tmp_path / "photos" / "*.jpg")])] == ["dog.jpg"]  # wildcard
    check_main([str(tmp_path / "m.pt"), str(tmp_path / "photos")])
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2 and lines[0].startswith("dog.jpg") and "food " in lines[0]

    values = [350.0, 234.0, 11.0, 9.0, 50.0]
    assert describe(values + [0.03], 0.5, True) == "food 0.03 -> NOT FOOD"
    assert describe(values + [0.97], 0.5, True).startswith("food 0.97 -> food | 350 kcal, mass 234 g")
    assert describe(values, 0.5, False).startswith("(model has no food output) 350 kcal")
