# Using the model in a React Native / Expo (Android) app

> **Untested on a device.** A working app now exists in the separate
> [calorie-app](https://github.com/Brandon-Lee-Last/calorie-app) repo (branch `claude/expo-app-v0`);
> use it instead of the sketch below. It type-checks against the real library types and bundles,
> but has not yet been built or run on an Android phone.

## The model file

`models/nutrition_v0.tflite` (4.45 MB) with `models/MODEL_CARD.md` (contract, training, accuracy,
limits) and `models/golden/` (a photo plus its expected output). Copy the `.tflite` into your Expo
app's assets. First thing to do in the app: run the golden photo through the model and confirm it
returns the values in `golden/example_expected.json`. That checks loading, tensor layout and output
order before you try any real photo.

## Model contract

Checked by converting a checkpoint with `export_tflite` and inspecting the file
with the LiteRT interpreter on a desktop CPU: input `[1, 224, 224, 3]` `uint8`,
output `[1, 5]` `float32`, 4.45 MB (fp32). Not yet run on a phone. Index the
tensors by position, since their auto-generated names are not meaningful.

| | |
|---|---|
| Input `image` | `uint8`, shape `[1, 224, 224, 3]`, RGB, NHWC, raw 0-255 pixels |
| Output `nutrition` | `float32`, shape `[1, 5]` = `[calories kcal, mass g, fat g, carb g, protein g]` |

Do **not** scale to [0, 1] or normalise in the app; the graph does it. The input
size comes from the checkpoint (`img_size`, default 224); the TFLite export fixes
batch size to 1.

## Setup (from the library's README)

```bash
npm install react-native-fast-tflite react-native-nitro-modules
```

`app.json`:

```json
{ "expo": { "plugins": [["react-native-fast-tflite", { "enableAndroidGpuLibraries": true }]] } }
```

Then `npx expo prebuild` and build a **development build**. The library is a
native module, so I expect Expo Go cannot load it (its README does not say).

`metro.config.js`: add `'tflite'` to `resolver.assetExts` so
`require('./assets/nutrition.tflite')` works.

## Sketch (checked against react-native-fast-tflite 3.0.1's types)

```ts
import { loadTensorflowModel } from 'react-native-fast-tflite';

// delegates is a required ARRAY: [] = plain CPU, ['android-gpu'] = GPU.
const model = await loadTensorflowModel(require('./assets/nutrition_v0.tflite'), []);
// rgb: Uint8Array of length 224 * 224 * 3, RGB interleaved. Pass its ArrayBuffer (exactly that size).
const [out] = await model.run([rgb.buffer]);          // run(input: ArrayBuffer[]): Promise<ArrayBuffer[]>
const [calories, mass, fat, carb, protein] = new Float32Array(out!);
```

`model.inputs` and `model.outputs` report each tensor's name, data type and shape, which the app uses to
check the loaded file against the contract. Delegates in the types: `'android-gpu'`, `'nnapi'`,
`'core-ml'`, `'metal'`. The library's README says NNAPI is deprecated on Android 15 and recommends the
GPU delegate; the model is small enough that plain CPU may be fast enough.

## Open spikes (not yet researched)

1. **Pixels from a photo.** Implemented in the app repo (`src/imaging/`): `expo-image-manipulator` decodes
   (capped at 1024 px) and squashes to 224x224, saves a lossless PNG, and a pure-JavaScript decoder
   (`fast-png`) turns it into raw RGB. The JavaScript decoding is tested to match PIL exactly on the golden
   photo; the manipulator step has not run on a device.
2. **Resize parity.** Training uses PIL bilinear resizing. Run the same photos
   through `python -m calorie_model.predict` and through the app; if outputs
   differ noticeably, either match the resampler in training or add
   resize-method augmentation.
3. **Latency and accuracy on real hardware**, CPU vs GPU delegate, fp32 vs int8.
