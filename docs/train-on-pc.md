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

## 3. Optional: one-minute speed check

```bash
python -m calorie_model.bench
```

Times a training step on random data for every combination of cuDNN autotuning,
channels-last layout and mixed precision, plus the GPU augmentation on its own. On an
RTX 3060 it showed channels-last making training about 5x slower (190-240 ms per step
vs 42 ms) and mixed precision giving no speedup, so both are off by default now
(mixed precision is opt-in with `--amp`). That puts a 30-epoch run at roughly six
minutes (estimate, not yet re-measured).

## 4. Train two runs and compare

```bash
python -m calorie_model.train --views overhead      --out runs/overhead_only
python -m calorie_model.train --views overhead,side --out runs/overhead_plus_side
```

Defaults: MobileNetV3-Small, ImageNet-pretrained (downloaded on first use), 224 px,
30 epochs, batch 64, fp32. Every image is decoded once at startup into
GPU memory (about 3.5 GB, roughly a minute; `--load-threads` sets the decoding
threads) and augmented on the GPU, so there are no DataLoader worker processes.
That replaced an earlier version that managed only ~82 images/s and crashed with
`WinError 6` on Windows. Out of GPU memory: lower `--batch-size`, or if the
startup step fails, close other GPU programs. Try `--backbone mobilenet_v3_large`
as a second experiment.

Each run writes `best.pt`, `history.json` and `test_metrics.json` in its folder.
The end of the log shows the test table: overall, per view type, and the
`baseline_median` row (always guess the training median). A real result must
beat that baseline clearly (about 185 kcal MAE on the full test set).

### More experiments (each takes minutes)

Compare runs on the `view_overhead` row (or the same view in both), not on `all`:
different runs can have different test sets. Measured seed-to-seed spread is about
1-2 kcal on calories (two runs), so gaps under ~3 kcal mean nothing.

```bash
python -m calorie_model.train --seed 1 --out runs/seed1                        # noise: repeat with seeds 1, 2
python -m calorie_model.train --backbone mobilenet_v3_large --out runs/large    # bigger model (12.8 MB fp32)
python -m calorie_model.train --epochs 60 --out runs/long                       # longer schedule
python -m calorie_model.train --train-fraction 0.25 --out runs/frac25           # learning curve: 25%, 50%, 100%
python -m calorie_model.train --train-fraction 0.5  --out runs/frac50           # of the training dishes
```

## Teaching the model what is NOT food (the "photo of a dog gets calories" fix)

A model trained only on food photos has no idea what a non-food photo is: it still prints five confident
numbers for a dog. This adds a sixth output, the probability that the photo shows food, trained on
Nutrition5k plus about 22,000 Open Images photos: roughly 9,000 real-world food photos and 13,000 non-food
photos (thousands of dogs, cats and people among them). The app then shows "this doesn't look like food"
instead of calories.

```bash
git pull
python -m calorie_model.prepare_openimages --out data/openimages
python -m calorie_model.train --openimages-root data/openimages --out runs/gate
```

* **Download:** about 23,000 images through Amazon's public mirror of Open Images, plus ~250 MB of label
  files; about 0.7 GB kept on disk afterwards. In my test it fetched ~50 images/s, but on a real run from a
  home connection it managed about 7 images/s and took about 54 minutes; plan for that. Safe to Ctrl+C
  and re-run. Quick test run: add `--scale 0.02`.
* **Training:** about 35,000 images go into GPU memory (roughly 5.5 GB; fits a 12 GB card). If it runs out
  of GPU memory, close other GPU programs, or add `--views overhead` to use far fewer Nutrition5k images.
  Measured: about 30-45 s per epoch after a 97 s startup, so a 30-epoch run takes about 20 minutes.
* **The result block to read** (printed at the end, under the TEST table):
  `food accepted` (real food photos it lets through), `non-food rejected` (photos it turns away, with
  separate lines for animals, people and everything else), the Nutrition5k photos recognised as food, and a
  threshold sweep. You want both of the first two high (the sweep shows the trade-off: a higher cut-off
  rejects more non-food but also turns away more real food). The nutrition numbers above it should stay
  about where they were.
* **Try your own photos**, the dog included. Make a folder of ~20 food photos and ~20 non-food photos taken
  with your phone, then:
  ```bash
  python -m calorie_model.check_photos runs/gate/best.pt phone_photos
  ```
  It prints each photo's food probability and either `NOT FOOD` or the estimate, using the same resize as
  the app. This is the test that matters: Open Images and Nutrition5k are not your phone.
* **Then send the checkpoint** as in section 5, and I will convert it to TFLite and update the app.

Photos used here are CC BY 2.0 (author and licence for each are in `data/openimages/index.csv`). Whether
training on them needs more than attribution is a question for someone qualified; this is not legal advice.

## 5. What to send back

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
