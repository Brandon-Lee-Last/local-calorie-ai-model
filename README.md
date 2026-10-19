# local-calorie-ai-model

A lightweight, fully on-device model that takes a photo of food and estimates
**calories, mass and macronutrients** (fat, carbs, protein). Nothing leaves the
phone.

Target: **Android**, inside a **React Native / Expo** app (see
[docs/android-expo.md](docs/android-expo.md)).

> Status: first real results exist (see **Results so far**). On the official Nutrition5k
> test dishes the model's calorie error is roughly 60-70% lower than always guessing
> the median. A trained model is exported as `models/nutrition_v0.tflite` (4.45 MB, see its model
> card); it matches PyTorch to under 0.001 kcal/g on 260 real photos on a desktop CPU. Nothing has
> been run on a phone, and none of this is tested on phone photos.

## Approach (v0)

One network, one forward pass:

```
RGB photo, resized to 224x224
  -> /255 and ImageNet normalisation (inside the graph)
  -> MobileNetV3-Small backbone (ImageNet-pretrained)
  -> global average pool -> Linear(576, 256) -> Hardswish -> Dropout
  -> Linear(256, 5) -> softplus
  -> [calories kcal, mass g, fat g, carb g, protein g]
```

- **Direct regression** on [Nutrition5k](https://github.com/google-research-datasets/nutrition5k)
  (overhead photos of real cafeteria dishes with weighed, dish-level labels).
  Predicting mass and macros jointly with calories gives the shared features
  more supervision than calories alone.
- **MobileNetV3-Small**: 1.08M parameters, 4.3 MB as fp32 ONNX, 1.3 MB after
  int8 quantisation (measured, untrained weights). `mobilenet_v3_large`
  (3.2M params, 12.8 MB fp32) is available via `--backbone`.
- **App-facing contract** (identical for ONNX and TFLite): input `image`
  `uint8 (N, 224, 224, 3)` RGB, NHWC, raw 0-255 pixels; output `nutrition`
  `float32 (N, 5)` in kcal/grams. All scaling, normalisation and the transpose
  live inside the graph, so the app only resizes and passes bytes. This removes
  the most common train/serve mismatch.

### Runtime for Expo on Android

The app-facing artifact is **TFLite/LiteRT** loaded with
[`react-native-fast-tflite`](https://github.com/mrousavy/react-native-fast-tflite),
chosen because its README documents an Expo config plugin and an Android GPU
delegate (it also says NNAPI is deprecated on Android 15 and recommends the GPU
delegate). ONNX stays as the reference format: it is what the Python tests
verify, and `onnxruntime-react-native` (which also documents an Expo config
plugin and supports `.onnx` models) is the fallback if TFLite conversion or
`fast-tflite` causes trouble. Caveats: `fast-tflite` is documented as
maintained "as is", and I could not confirm which execution providers
`onnxruntime-react-native` exposes on Android.

Either way this needs a **development build** (native module), not Expo Go.
A 1M-parameter model should be fast on CPU alone, so the GPU delegate is an
optimisation to measure, not a requirement.

### Why not classify-then-lookup?

Food classification (e.g. Food-101) plus a nutrition table (e.g. USDA FoodData
Central) times an estimated portion is more interpretable and scales to far
more foods, because class-labelled photos are plentiful while photos with
weighed nutrition labels are scarce. It still needs a portion estimator, which
is the hard part. Since the goal is broad food coverage, this hybrid is the
planned v1: classifier for per-gram nutrient density, regressor for portion.

## Layout

```
src/calorie_model/
  model.py          NutritionNet, target order and scales
  prepare_data.py   download Nutrition5k over HTTPS, build a compact 256px image cache
  prepare_openimages.py  food / not-food photos from Open Images (CC BY 2.0) for the food output
  sa_foods.py       South African dish list + per-100 g sheet loader, weighed-plate arithmetic
  prepare_commons.py  openly licensed South African dish photos from Wikimedia Commons (untested live)
  ingest_meals.py   weighed phone photos of meals -> training folder (--extra-data)
  check_photos.py   run a checkpoint on your own photos exactly as the app will
  evaluate_checkpoint.py  accuracy AND consistency (same dish, different angles), plain vs averaged
  data.py           cached-dataset loader (train/val/test), augmentations, synthetic data
  train.py          training loop (AdamW + OneCycle), val split, final test report + baseline
  export.py         checkpoint -> ONNX (+ parity check, optional int8)
  export_tflite.py  checkpoint -> .tflite via litert-torch (+ parity check)
  predict.py        run an ONNX model on one image, print JSON
docs/adding-data.md   how to plug in more labelled data, and the candidate sources I found
docs/south-african-foods.md   what the model does and does not know about South African food, and the plan
datasets/south_african/   dish list (draft) and an empty per-100 g sheet
docs/train-on-pc.md   step-by-step runbook for training on a Windows/Linux GPU machine
docs/android-expo.md
tests/test_pipeline.py, tests/test_south_african.py
```

## Quickstart

```bash
pip install -e ".[dev]"

# 1. Check the pipeline works without any data (does not measure accuracy)
pytest
python -m calorie_model.train --synthetic --no-pretrained --epochs 2 --out runs/smoke

# 2. Real training (full walkthrough in docs/train-on-pc.md)
python -m calorie_model.prepare_data --out data/nutrition5k      # needs `pip install av`; no gsutil
python -m calorie_model.train --data-root data/nutrition5k --epochs 30 --out runs/v0

# 3. Export
python -m calorie_model.export runs/v0/best.pt --out build/nutrition.onnx   # add --quantize
python -m calorie_model.predict build/nutrition.onnx some_food.jpg

# 4. TFLite for the app (separate env: pip install litert-torch ai-edge-litert, Python 3.11)
python -m calorie_model.export_tflite runs/v0/best.pt --out build/nutrition.tflite
```

Training picks the best epoch on a validation split of whole dishes held out from
the training pool, then reports the official test split once, per view type and
next to a baseline that always predicts the training median.

## Data

| Dataset | Use | Notes |
|---|---|---|
| Nutrition5k | v0 training: images + weighed calories/mass/macros | CC BY 4.0 per its README (commercial use allowed, cite the CVPR 2021 paper). Collected in a few California cafeterias, so expect bias. Bucket layout and the `rgb_{train,test}_ids.txt` split files are verified. 5,006 labelled dishes; only 3,490 have an overhead photo (1.45 GB). The side-angle videos (192 GB, up to 4 cameras per dish, many missing) are sampled: `prepare_data` fetches only the first ~1 MB of each and keeps one frame, giving ~4 images per dish. Those frames have a yellow cast and unusual angles, so they are more data, not phone-like data. |
| Open Images (food / not-food) | the food output: real-world food photos and many kinds of non-food | Photos are CC BY 2.0 per the dataset's metadata (author and licence kept per image in `index.csv`); labels CC BY 4.0. About 22,000 photos: ~9k food, ~13k non-food (dogs, cats, people, everyday scenes). Labels are human-verified but not exhaustive, so a few "non-food" photos may contain unlabelled food; images with plates, bowls, drinks or kitchens but no food label are left out of both groups. Whether training on CC BY images needs more than attribution is a question for someone qualified. |
| Food-101 and larger food-class sets | v1 classifier for broad coverage | Licences vary per dataset and have **not** been checked yet; do that before training on any of them. |
| USDA FoodData Central / Open Food Facts | per-100 g nutrient tables for the v1 lookup | Licence terms to be confirmed. |

## Results so far

MobileNetV3-Small, ImageNet-pretrained, 224 px, 30 epochs, one run each (seed 0), trained
on an RTX 3060. MAE = mean absolute error; the baseline always guesses the training median.

| Run | Evaluated on | Calorie MAE | Mass MAE | Baseline calorie MAE |
|---|---|---|---|---|
| overhead only (2,689 train images) | 507 overhead test photos | **52.3 kcal (20.5%)** | 35.6 g | 169.7 kcal |
| overhead + side (16,969 train images) | the same 507 overhead photos | **52.7 kcal (20.6%)** | 34.7 g | 169.7 kcal |
| overhead + side | ~2,600 side-camera test frames | **76.2 kcal (29.9%)** | 45.9 g | not shown |
| overhead + side | all 3,110 test images | 72.4 kcal (28.4%) | 44.1 g | 185.6 kcal |

How to read this:

- **Compare like with like.** The overhead-only and overhead+side models should be compared on the
  same 507 overhead photos (rows 1 and 2): 52.3 vs 52.7 kcal. That is no measurable difference.
  Adding the side frames did not help or hurt overhead accuracy; it lets the model handle side views at
  all. The "all images" row is a different, harder test set (it includes side frames), so it cannot be
  compared with the overhead-only run.
- **Validation looks better than test** by roughly 27-32 kcal in every run below. Trust the test number.
  A likely cause is repeated menu items making held-out training dishes easier than the official test
  dishes, but that is a guess, not checked.
- **These are cafeteria camera-rig photos, not phone photos.** How the model does on real phone photos
  is unknown and is the most important thing still to measure.

### Noise, model size and amount of data

All on the same 3,110 official test images (calorie MAE in kcal / mass MAE in g). Fat, carb and
protein errors are in each run's `test_metrics.json` and have not been looked at yet.

| Run | All images | Overhead photos | Side frames |
|---|---|---|---|
| small, seed 0 | 72.4 / 44.1 | 52.7 / 34.7 | 76.2 / 45.9 |
| small, seed 1 | 74.1 / 44.5 | 52.9 / 36.8 | 78.3 / 46.0 |
| large (3.2M params, 12.8 MB fp32) | 72.9 / 40.6 | 52.3 / 31.5 | 77.0 / 42.4 |
| small, 50% of the training dishes | 85.2 / 47.8 | 61.4 / 41.0 | 89.8 / 49.1 |
| always guess the median | 185.6 / 117.7 | | |

- **Noise:** two seeds differ by about 1.7 kcal on all images (2 runs, so a rough estimate). Gaps under
  roughly 3 kcal should be read as no difference.
- **Bigger model:** calories are no better (72.9 vs 72.4 and 74.1), mass is about 8% better (40.6 vs
  ~44.3 g), at 2.2x the epoch time and 3x the file size. Its training loss ended far lower (0.021 vs
  0.043) with the same test error, so it memorises more without generalising more: the limit looks like
  data, not model capacity.
- **More data:** halving the training dishes raised calorie error about 16% (85.2 vs ~73.3), so doubling
  the dishes should lower it by about 14%. That comes from two points; the shape beyond them is
  unknown, and there are only 4,297 training dishes, so going further needs new data.

### Food / not-food results

First full run: Nutrition5k + Open Images (34,969 training images), MobileNetV3-Small, 30 epochs, one seed,
RTX 3060 (about 30-45 s per epoch). Checked on 3,000 Open Images photos from the dataset's validation subset,
which were never in training. Accepted as food if the probability is at least 0.5.

| | |
|---|---|
| Real food photos accepted | 91.5% (1,000 photos) |
| Non-food photos rejected | 96.5% overall (2,000 photos) |
| ... dogs and cats | 98.8% rejected (500 photos) |
| ... people | 98.2% rejected (500 photos) |
| ... everyday scenes | 94.5% rejected (1,000 photos) |
| Nutrition5k test photos recognised as food | 100% |
| AUC | 0.983 |
| Threshold 0.2 -> 0.8 | food accepted 92.7% -> 89.7%, non-food rejected 95.8% -> 97.1% |

- The threshold barely matters (the model is confident), so 0.5 stays.
- 8.5% of "food" photos are turned away. Some may be label noise (Open Images tags such as "Cuisine" or
  "Fruit" also cover images that are not meals), but I have not looked at which ones, so that is a guess.
  The app's "Estimate anyway" button covers false rejections.
- **Cost to the nutrition outputs:** calorie error on all test images is 77.5 kcal, against 72.4 and 74.1
  for the two earlier runs without the food output (overhead photos 53.5 vs about 52.8; side frames 82.1 vs
  about 77.3). Roughly 5% worse, mostly on the rig's side frames. One run, so part may be noise.
- Not yet tested on phone photos. That is the test that matters; use `check_photos`.

## Known gap: it will "estimate" anything

The first model (v0) was trained only on food, so a photo of a dog still gets five confident-looking
numbers (a real phone test returned about 200 kcal). The code now has a food / not-food output to fix this:
a sixth output, the probability the photo shows food, trained on Nutrition5k plus about 22,000 Open Images
photos (real-world food, and non-food including dogs, cats and people). The app shows "this doesn't look
like food" when it is low. A first full-size run worked (see "Food / not-food results" below); the model
is not yet converted and shipped in the app, so until it is, the app still has no way to refuse a non-food
photo. See `docs/train-on-pc.md` ("Teaching the model what is NOT food").

## South African food

**The model does not know South African food yet.** Nothing it was trained on is South African, and no
public dataset of South African dishes with weights or nutrition exists that I could find. The tooling to
change that is in place (a draft list of 57 dishes, a Wikimedia Commons downloader, a weighed-meal
converter, a held-out test that compares a run with and without South African data); the data is not. Whether
it "knows" South African food will be a measured number, not a slogan. See
[docs/south-african-foods.md](docs/south-african-foods.md).

## Limitations to be honest about

- **A single photo cannot determine calories exactly.** Portion size, depth,
  oil/sugar/sauce and hidden ingredients are not recoverable from pixels. The
  app should show an estimate or range, not a precise number.
- **Domain gap.** Nutrition5k is overhead shots of cafeteria dishes under
  controlled lighting. Phone photos at an angle, other cuisines and packaged
  food will be worse until we add data that matches real usage.
- **Resize mismatch.** Training resizes with PIL (bilinear); the app will resize
  with an Android library. Different resamplers shift pixels slightly. Compare
  device and Python outputs on the same photos before trusting any accuracy number.
- This is an estimation aid, not medical or dietary advice.

## Decisions so far

| Question | Answer |
|---|---|
| Platform | Android only, React Native + Expo |
| Food scope | As broad as possible → plan the classifier + nutrient-table hybrid (v1) |
| Outputs | Calories + macros (fat, carbs, protein) |
| Portion/scale cue | Open to whatever is needed; see roadmap step 2 |

## Roadmap

1. **Done: train v0, seed noise, larger model, half-data run** (see Results so far). Next: the 25%
   point of the learning curve, and a look at the fat/carb/protein errors.
2. **Measure whether depth is worth it.** Nutrition5k includes overhead depth for
   a subset of dishes. Train an RGB-D variant and compare against RGB-only on the
   same dishes before building any depth capture in the app (Android depth
   support depends on the device and has not been investigated yet).
3. **Verify TFLite on a device:** conversion and desktop-CPU parity are done
   (4.45 MB fp32, uint8 NHWC in, float32 `[1, 5]` out). Still to do: int8/fp16
   TFLite (not attempted; the ONNX int8 path is separate), then latency and the
   accuracy drop on a real mid-range Android phone.
4. **Broaden coverage:** classification head + nutrient lookup (v1), after
   checking dataset licences.
5. **Close the domain gap:** collect phone-style photos; add an uncertainty output
   so the app can show ranges.
6. Optional scale cues in the app (reference object, depth) if step 2 says they help.
