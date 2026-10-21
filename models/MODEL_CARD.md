# nutrition_v1.tflite

Estimates calories, mass and macros from one food photo, and says how likely it is that the photo shows food
at all. Fully on-device, 4.42 MB.

> Trained on cafeteria photos plus generic Open Images photos. **It has never seen South African food, and it
> is not yet tested on a phone or on phone photos.** Treat outputs as rough estimates, not measurements, and
> not medical or dietary advice.

## Contract

| | |
|---|---|
| Input | 1 tensor, `uint8`, shape `[1, 224, 224, 3]`, RGB, NHWC, raw 0-255 pixels. The photo is **squashed** to 224 x 224 (no cropping, aspect ratio not preserved). No other preprocessing: scaling and normalisation are inside the model. |
| Output | 1 tensor, `float32`, shape `[1, 6]`, in this order and units: `calories` (kcal), `mass` (g), `fat` (g), `carb` (g), `protein` (g), `food` (probability 0-1 that the photo shows food). The first five are always >= 0. |
| Runtime | LiteRT / TFLite. Index the tensors by position; their auto-generated names are meaningless. |

Treat the five nutrition values as meaningless when `food` is below the threshold (0.5 below). `nutrition_v1_tta.tflite`
has the identical contract: it runs the photo mirrored, flipped and both and averages the four answers inside
the model (about 4x the compute, desktop CPU 11.3 ms against 2.9 ms), so the same photo gives a steadier
answer. Its accuracy has **not** been measured here; the numbers below are for the plain model.

## Golden test

`golden/example_224.png` is an already-resized 224 x 224 photo. Feeding its exact pixels to the model
must give the values in `golden/example_expected.json` (`expected`, `expected_food`; within ~0.5 kcal/g).
This proves model loading, tensor layout and output order in your app. It checks that the app reproduces the
model, not that the model is accurate (the expected values are the model's own, and differ from the true
label shown there).

`golden/example_256.jpg` is the same photo before resizing. Let the app resize it to 224 x 224 itself and
compare with the expected values: this tests resize parity. Small differences are normal (different
resamplers); large ones mean the app's preprocessing needs matching to training, which squashes with
PIL bilinear resizing.

`gradient` in the JSON describes a synthetic image (a formula, no file) whose food probability must come out
below the threshold: it checks that the food output can say "no" on the device. It is not a food / not-food
accuracy test.

## Where it came from

- Weights: `runs/gate/best.pt` on branch `checkpoints-v1` (commit `5b80cfa`), converted with
  `python -m calorie_model.export_tflite` (litert-torch). Matches PyTorch to 4.8e-4 on random pixels
  (plain) and 1.5e-4 (four-flip) on a desktop CPU.
- Architecture: MobileNetV3-Small (ImageNet-pretrained) with a 5-output regression head and a food logit,
  1.08M parameters.
- Training: Nutrition5k (overhead photos plus one frame per side camera) together with about 22,000 Open Images
  photos for the food output (34,969 training images in total), 224 px, 30 epochs, AdamW + OneCycle,
  fp32, one seed, RTX 3060. The checkpoint was chosen on clean Nutrition5k validation.

## Accuracy

Mean absolute error per dish on the official Nutrition5k test split (3,110 images; 507 overhead, 2,603
side-camera frames). "Median" is a baseline that always predicts the training median.

| | calories kcal | mass g | fat g | carb g | protein g |
|---|---|---|---|---|---|
| all test images | **77.5** (30%) | 46.0 | 6.1 | 7.3 | 5.4 |
| overhead photos | 53.5 (21%) | 35.8 | 3.9 | 6.2 | 5.2 |
| side-camera frames | 82.1 (32%) | 48.0 | 6.5 | 7.5 | 5.4 |
| median baseline (all) | 185.6 (73%) | 117.7 | 11.6 | 14.0 | 13.7 |

Compared with `nutrition_v0` (74.1 kcal on all images) calorie error is about 5% worse, mostly on the side
frames; one run each, so part of that may be noise. Seed-to-seed spread was roughly 1-2 kcal for v0.

### Food / not-food

Checked on 3,000 Open Images photos from the dataset's validation subset, never in training. Accepted as food
at probability >= 0.5.

| | |
|---|---|
| Real food photos accepted | 91.5% (1,000 photos) |
| Non-food rejected | 96.5% overall (2,000 photos) |
| ... dogs and cats | 98.8% rejected (500) |
| ... people | 98.2% rejected (500) |
| ... everyday scenes | 94.5% rejected (1,000) |
| Nutrition5k test photos recognised as food | 100% |
| AUC | 0.983 |

Raising the threshold from 0.2 to 0.8 moves food accepted from 92.7% to 89.7% and non-food rejected from 95.8%
to 97.1%, so 0.5 is kept. About 8.5% of real food photos are turned away.

## Limits

- **Domain:** Google cafeteria dishes under rig lighting, and Open Images for the food output. Phone photos,
  other cuisines, packaged food, drinks and unusual plates are untested and likely worse.
- **South African food:** none was used in training and none has been measured. It may estimate badly, and it
  may reject some South African dishes as "not food". Do not describe it as knowing South African food.
- **A photo cannot show** oil, sugar, hidden ingredients or depth, so there is an irreducible error. Fat
  is probably the weakest output. Show ranges or "about", not exact numbers.
- **Resize:** the app's resizing differs slightly from training's (see the golden test).

## Attribution

Trained on [Nutrition5k](https://github.com/google-research-datasets/nutrition5k) (Google Research,
CC BY 4.0): Thames et al., "Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food",
CVPR 2021, and on photos from [Open Images](https://storage.googleapis.com/openimages/web/index.html)
(CC BY 2.0 images, CC BY 4.0 labels; the author and licence of each photo are kept in the dataset's
`index.csv` when it is prepared). Credit both in the app. Whether the licences need more than that for a
derived model is a question for someone qualified; this is not legal advice.
