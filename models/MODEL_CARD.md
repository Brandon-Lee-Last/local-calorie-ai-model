# nutrition_v0.tflite

Estimates calories, mass and macros from one food photo. Fully on-device, 4.45 MB.

> Trained on cafeteria photos only. **Not yet tested on a phone or on phone photos.**
> Treat outputs as rough estimates, not measurements, and not medical or dietary advice.

## Contract

| | |
|---|---|
| Input | 1 tensor, `uint8`, shape `[1, 224, 224, 3]`, RGB, NHWC, raw 0-255 pixels. The photo is **squashed** to 224 x 224 (no cropping, aspect ratio not preserved). No other preprocessing: scaling and normalisation are inside the model. |
| Output | 1 tensor, `float32`, shape `[1, 5]`, in this order and units: `calories` (kcal), `mass` (g), `fat` (g), `carb` (g), `protein` (g). Always >= 0. |
| Runtime | LiteRT / TFLite. Index the tensors by position; their auto-generated names are meaningless. |

## Golden test

`golden/example_224.png` is an already-resized 224 x 224 photo. Feeding its exact pixels to the model
must give the values in `golden/example_expected.json` (within ~0.5 kcal/g). This proves model loading,
tensor layout and output order in your app. It checks that the app reproduces the model, not that the
model is accurate (the expected values are the model's own, and differ from the true label shown there).

`golden/example_256.jpg` is the same photo before resizing. Let the app resize it to 224 x 224 itself and
compare with the expected values: this tests resize parity. Small differences are normal (different
resamplers); large ones mean the app's preprocessing needs matching to training, which squashes with
PIL bilinear resizing.

## Where it came from

- Weights: `runs/seed1/best.pt` on branch `checkpoints-v0` (commit `d82eb51`), converted with
  `python -m calorie_model.export_tflite` (litert-torch). Output matches PyTorch to < 0.001 kcal/g on
  260 real photos.
- Architecture: MobileNetV3-Small (ImageNet-pretrained) with a 5-output regression head, 1.08M parameters.
- Training: Nutrition5k, overhead photos plus one frame per side camera (16,969 training images from the
  official training dishes plus unsplit extras), 224 px, 30 epochs, AdamW + OneCycle, batch 64, fp32, seed 1.
  Code at commit `bedc30c`.

## Accuracy

Mean absolute error per dish on the official Nutrition5k test split (3,110 images; 507 overhead, 2,603
side-camera frames). "Median" is a baseline that always predicts the training median.

| | calories kcal | mass g | fat g | carb g | protein g |
|---|---|---|---|---|---|
| all test images | **74.1** (29%) | 44.5 | 5.9 | 6.9 | 5.2 |
| overhead photos | 52.9 (21%) | 36.8 | 4.2 | 5.6 | 5.6 |
| side-camera frames | 78.3 (31%) | 46.0 | 6.3 | 7.2 | 5.1 |
| median baseline (all) | 185.6 (73%) | 117.7 | 11.6 | 14.0 | 13.7 |

Seed-to-seed spread is roughly 1-2 kcal (two runs). On photos it trained on, calorie error is about
16 kcal, so the model memorises training dishes: expect worse on food unlike the training set.

## Limits

- **Domain:** Google cafeteria dishes under rig lighting. Phone photos, other cuisines, packaged food,
  drinks and unusual plates are untested and likely worse.
- **A photo cannot show** oil, sugar, hidden ingredients or depth, so there is an irreducible error. Fat
  is probably the weakest output. Show ranges or "about", not exact numbers.
- **Resize:** the app's resizing differs slightly from training's (see the golden test).

## Attribution

Trained on [Nutrition5k](https://github.com/google-research-datasets/nutrition5k) (Google Research,
CC BY 4.0): Thames et al., "Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food",
CVPR 2021. Credit it in the app. Whether the licence needs more than that for a derived model is a
question for someone qualified; this is not legal advice.
