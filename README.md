# local-calorie-ai-model

A lightweight, fully on-device model that takes a photo of food and estimates
**calories, mass and macronutrients** (fat, carbs, protein). Nothing leaves the
phone.

Target: **Android**, inside a **React Native / Expo** app (see
[docs/android-expo.md](docs/android-expo.md)).

> Status: v0 scaffold. The train → export → inference pipeline runs end to end
> and is tested (`pytest`). The TFLite export converts and matches PyTorch to
> ~4e-6 when run on a desktop CPU. **No real model has been trained yet**, so
> there are no accuracy numbers, and nothing has been run on a phone.

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
  data.py           Nutrition5k loader, augmentations, synthetic smoke-test data
  train.py          training loop (AdamW + OneCycle, weighted SmoothL1), eval in real units
  export.py         checkpoint -> ONNX (+ parity check, optional int8)
  export_tflite.py  checkpoint -> .tflite via litert-torch (+ parity check)
  predict.py        run an ONNX model on one image, print JSON
scripts/download_nutrition5k.sh
docs/android-expo.md
tests/test_pipeline.py
```

## Quickstart

```bash
pip install -e ".[dev]"

# 1. Check the pipeline works without any data (does not measure accuracy)
pytest
python -m calorie_model.train --synthetic --no-pretrained --epochs 2 --out runs/smoke

# 2. Real training
scripts/download_nutrition5k.sh data/nutrition5k      # needs gsutil
python -m calorie_model.train --data-root data/nutrition5k --epochs 30 --out runs/v0

# 3. Export
python -m calorie_model.export runs/v0/best.pt --out build/nutrition.onnx   # add --quantize
python -m calorie_model.predict build/nutrition.onnx some_food.jpg

# 4. TFLite for the app (separate env: pip install litert-torch ai-edge-litert, Python 3.11)
python -m calorie_model.export_tflite runs/v0/best.pt --out build/nutrition.tflite
```

Training reports MAE per target in real units and calorie MAE as a percentage
of the mean true calories.

## Data

| Dataset | Use | Notes |
|---|---|---|
| Nutrition5k | v0 training: images + weighed calories/mass/macros | Released under CC BY 4.0 per its README (commercial use allowed, cite the CVPR 2021 paper). Collected in a few California cafeterias, so expect bias. The README does not list the split filenames; the loader accepts `rgb_<split>_ids.txt` or a single `rgb_<split>*.txt` and reports what it found otherwise. The split files and `rgb.png` paths are still unchecked against a real download. |
| Food-101 and larger food-class sets | v1 classifier for broad coverage | Licences vary per dataset and have **not** been checked yet; do that before training on any of them. |
| USDA FoodData Central / Open Food Facts | per-100 g nutrient tables for the v1 lookup | Licence terms to be confirmed. |

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

1. **Train v0** on Nutrition5k; record baseline error and compare against always
   predicting the dataset mean.
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
