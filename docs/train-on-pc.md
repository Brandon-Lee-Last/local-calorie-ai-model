# Training v0 on a GPU machine (Windows or Linux)

Written for an NVIDIA GPU with 12 GB (e.g. RTX 3060), a 6-core CPU and 16 GB RAM.
The data tool and training code are tested on Linux CPU only; nothing here has
yet run on a GPU or on Windows, so expect to fix small things and tell me what
broke.

## 1. Set up

1. Install **Python 3.11 or 3.12** and **Git**.
2. Get the code and make an environment:

   ```bash
   git clone https://github.com/Brandon-Lee-Last/local-calorie-ai-model
   cd local-calorie-ai-model
   git checkout claude/quirky-brown-36reqx
   python -m venv .venv
   .venv\Scripts\activate          # Windows;  Linux/macOS: source .venv/bin/activate
   ```
3. Install **PyTorch with CUDA first**, using the command from the selector at
   https://pytorch.org/get-started/locally/ (I can't check which CUDA build is
   current). Then the rest:

   ```bash
   pip install -e ".[data,export]" pytest
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   This must print `True` and your GPU name before you continue.
4. Optional sanity check: `pytest` (the TFLite test skips; that is expected).

## 2. Download and cache the data

```bash
python -m calorie_model.prepare_data --out data/nutrition5k
```

* Plain HTTPS from Google's public bucket; no account or `gsutil`.
* About 20 GB is downloaded in total but only ~0.5 GB is kept (256 px JPEGs). On
  a 4-core Linux box a 60-dish sample took 20 s, which would put the full run at
  roughly half an hour; your connection decides the real time.
* Safe to Ctrl+C and re-run; finished files are skipped. Failures are listed in
  `data/nutrition5k/failures.txt`; just re-run to retry them.
* Prints the final image counts per view. Expect on the order of 20k images.
* Overhead photos only (much faster, 1.5 GB): add `--views overhead`.

## 3. Train two runs and compare

```bash
python -m calorie_model.train --views overhead      --out runs/overhead_only
python -m calorie_model.train --views overhead,side --out runs/overhead_plus_side
```

Defaults: MobileNetV3-Small, ImageNet-pretrained (downloaded on first use), 224 px,
30 epochs, batch 64, mixed precision, `--num-workers` = min(8, CPU threads). If
data loading errors on Windows, retry with `--num-workers 0`. Out of GPU memory:
lower `--batch-size`. Try `--backbone mobilenet_v3_large` as a second experiment.

Each run writes `best.pt`, `history.json` and `test_metrics.json` in its folder.
The end of the log shows the test table: overall, per view type, and the
`baseline_median` row (always guess the training median). A real result must
beat that baseline clearly (about 185 kcal MAE on the full test set).

## 4. What to send back

Paste the final `TEST (...)` table from both runs. If you want the TFLite file
made, push the checkpoint on a separate branch (checkpoints are git-ignored, so
force-add them; they are ~4 MB):

```bash
git checkout -b checkpoints-v0
git add -f runs/overhead_plus_side/best.pt runs/overhead_plus_side/test_metrics.json runs/overhead_plus_side/history.json
git commit -m "v0 checkpoint" && git push -u origin checkpoints-v0
```

TFLite conversion (`litert-torch`) lists Linux as its supported OS, so I will do
that step in the Linux environment rather than ask you to set it up on Windows.
ONNX export works anywhere: `python -m calorie_model.export runs/<run>/best.pt`.
