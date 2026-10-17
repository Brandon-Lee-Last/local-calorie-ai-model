# local-calorie-ai-model

A lightweight, fully on-device model that takes a photo of food and estimates
**calories, mass and macronutrients** (fat, carbs, protein). Nothing leaves the
phone.

> Status: v0 scaffold. The training / export / inference pipeline runs end to
> end, but **no real model has been trained yet**, so there are no accuracy
> numbers. Everything under "Targets" is a goal, not a measurement.

## Approach (v0)

One network, one forward pass:

```
RGB image (224x224, [0,1])
  -> ImageNet normalisation (inside the graph)
  -> MobileNetV3-Small backbone (ImageNet-pretrained)
  -> global average pool -> Linear(576, 256) -> Hardswish -> Dropout
  -> Linear(256, 5) -> softplus
  -> [calories kcal, mass g, fat g, carb g, protein g]
```

- **Direct regression** on [Nutrition5k](https://github.com/google-research-datasets/nutrition5k)
  (overhead photos of real cafeteria dishes with weighed, dish-level labels).
  Predicting mass and macros jointly with calories gives the shared features
  more supervision than calories alone.
- **MobileNetV3-Small** is ~1.5M parameters with the head (a test guards it
  under 5M), is hardware-friendly, and has pretrained weights in torchvision.
  `mobilenet_v3_large` is available via `--backbone` if accuracy matters more
  than size.
- Normalisation lives **inside** the exported graph and outputs are already in
  kcal / grams, so the mobile app only resizes to a square and scales pixels to
  `[0, 1]`. That removes the most common source of train/serve mismatch.
- Export is ONNX, with optional static int8 quantisation. ONNX Runtime Mobile
  runs on both iOS and Android; converting onward to TFLite / Core ML is a
  later step once the target platform is decided.

### Why not classify-then-lookup?

The alternative is food classification (e.g. Food-101) plus a nutrition table
(e.g. USDA FoodData Central) multiplied by an estimated portion. It is more
interpretable and covers many more foods, but needs a portion estimator
anyway, which is the hard part. It is a good **v1**, ideally combined with this
model: use the class for the per-gram nutrient density and the regressor for
portion size.

## Layout

```
src/calorie_model/
  model.py     NutritionNet, target order and scales
  data.py      Nutrition5k loader, augmentations, synthetic smoke-test data
  train.py     training loop (AdamW + OneCycle, weighted SmoothL1), eval in real units
  export.py    checkpoint -> ONNX (+ parity check vs PyTorch, optional int8)
  predict.py   run an ONNX model on one image, print JSON
scripts/download_nutrition5k.sh
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

# 3. Export and try it
python -m calorie_model.export runs/v0/best.pt --out build/nutrition.onnx
python -m calorie_model.predict build/nutrition.onnx some_food.jpg
```

Training reports MAE per target in real units and calorie MAE as a percentage
of the mean true calories.

## Targets (not yet measured)

| | goal |
|---|---|
| Model file | < 10 MB (fp32), < 3 MB (int8) |
| Latency | < 100 ms on a mid-range phone |
| Calorie error | to be set against a baseline once v0 is trained |

## Limitations to be honest about

- **A single photo cannot determine calories exactly.** Portion size, depth,
  oil/sugar/sauce and hidden ingredients are not recoverable from pixels. Expect
  a wide error band; the app should show an estimate or range, not a precise
  number.
- **Domain gap.** Nutrition5k is overhead shots of cafeteria dishes under
  controlled lighting. Phone photos taken at an angle, other cuisines and
  packaged food will be worse until we add data that matches real usage.
- This is an estimation aid, not medical or dietary advice.

## Open decisions

These shape the next steps and are the things worth deciding early:

1. **Target platform:** iOS, Android, or both (decides TFLite vs Core ML vs ONNX Runtime Mobile).
2. **Food scope:** Western cafeteria-style meals only, or specific cuisines / packaged foods.
3. **Outputs:** is calories + macros enough, or do we want ingredients / micronutrients?
4. **Scale cue:** can the app use depth/LiDAR or a reference object (coin, plate size) to improve portion estimates?
5. **Data licensing:** confirm the licence terms of Nutrition5k (and any added datasets) allow our intended use before shipping a model trained on them.

## Roadmap

1. Train v0 on Nutrition5k, record baseline error, compare against predicting the dataset mean.
2. Measure int8 accuracy drop and on-device latency on real hardware.
3. Add a food-classification head / auxiliary dataset (Food-101) and a USDA nutrient lookup.
4. Collect phone-style photos to close the domain gap; add uncertainty output.
5. Platform-specific conversion and a minimal demo app.
