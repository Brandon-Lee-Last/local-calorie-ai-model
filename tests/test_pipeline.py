import pytest
import torch

from calorie_model import TARGETS, NutritionNet
from calorie_model.data import SyntheticDataset
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
