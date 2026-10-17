import numpy as np
import pytest
import torch
from PIL import Image

from calorie_model import TARGETS, NutritionNet
from calorie_model.data import Nutrition5kDataset, SyntheticDataset, load_rgb_uint8
from calorie_model.model import count_parameters
from calorie_model.train import main as train_main

IMG = 96  # small so tests stay fast


@pytest.fixture(scope="module")
def net():
    return NutritionNet(pretrained=False).eval()


def test_output_shape_and_nonnegative(net):
    out = net(torch.rand(3, 3, IMG, IMG))
    assert out.shape == (3, len(TARGETS))
    assert (out >= 0).all()


def test_stays_lightweight():
    # Phone budget guard: the default model must stay well under 5M parameters.
    assert count_parameters(NutritionNet(pretrained=False)) < 5_000_000


def test_unknown_backbone_rejected():
    with pytest.raises(ValueError):
        NutritionNet("resnet9000", pretrained=False)


def test_synthetic_training_runs_and_checkpoints(tmp_path):
    train_main(
        ["--synthetic", "--no-pretrained", "--epochs", "2", "--batch-size", "16",
         "--img-size", str(IMG), "--num-workers", "0", "--device", "cpu", "--out", str(tmp_path)]
    )
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "history.json").exists()


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
        loss = torch.nn.functional.smooth_l1_loss(model(x), y, beta=0.1)
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
    assert list(result) == list(TARGETS)
    assert all(v >= 0 for v in result.values())


def test_load_rgb_uint8_shape_and_dtype(tmp_path):
    Image.new("RGB", (50, 80), (1, 2, 3)).save(tmp_path / "a.png")
    arr = load_rgb_uint8(tmp_path / "a.png", IMG)
    assert arr.shape == (IMG, IMG, 3) and arr.dtype == np.uint8


def _fake_nutrition5k(root, split_name="rgb_train_ids.txt"):
    (root / "metadata").mkdir(parents=True)
    (root / "dish_ids" / "splits").mkdir(parents=True)
    # Real rows are headerless, with a variable number of per-ingredient columns.
    (root / "metadata" / "dish_metadata_cafe1.csv").write_text(
        "dish_1,300.0,250.0,10.0,40.0,12.0,2,ingr_1,rice,100,130,0.3,28,2.5,ingr_2,egg,50,70,5,0.5,6\n"
        "dish_2,120.0,90.0,3.0,15.0,5.0,0\n"
        "dish_noimage,50.0,50.0,1.0,1.0,1.0,0\n"
    )
    (root / "dish_ids" / "splits" / split_name).write_text("dish_1\ndish_2\ndish_noimage\n")
    for d in ("dish_1", "dish_2"):
        img_dir = root / "imagery" / "realsense_overhead" / d
        img_dir.mkdir(parents=True)
        Image.new("RGB", (64, 48), (10, 20, 30)).save(img_dir / "rgb.png")


def test_nutrition5k_loader_parses_labels_and_skips_missing_images(tmp_path):
    _fake_nutrition5k(tmp_path)
    ds = Nutrition5kDataset(tmp_path, "train", img_size=IMG)
    assert len(ds) == 2  # dish_noimage has a label but no image
    x, y = ds[0]
    assert x.shape == (3, IMG, IMG) and 0 <= x.min() and x.max() <= 1
    # Targets are stored in scaled units: 300 kcal / 500 = 0.6, 250 g / 400 = 0.625.
    assert torch.allclose(y[:2], torch.tensor([0.6, 0.625]), atol=1e-6)


def test_nutrition5k_loader_tolerates_other_split_filename(tmp_path):
    _fake_nutrition5k(tmp_path, split_name="rgb_train_splits.txt")
    assert len(Nutrition5kDataset(tmp_path, "train", img_size=IMG)) == 2


def test_nutrition5k_loader_error_lists_found_split_files(tmp_path):
    _fake_nutrition5k(tmp_path, split_name="weird.txt")
    with pytest.raises(FileNotFoundError, match="weird.txt"):
        Nutrition5kDataset(tmp_path, "train", img_size=IMG)


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
    assert out.shape == (1, len(TARGETS)) and np.isfinite(out).all()
