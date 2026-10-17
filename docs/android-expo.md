# Using the model in a React Native / Expo (Android) app

> **Untested on a device.** This is assembled from the `react-native-fast-tflite`
> README and the model contract in this repo. Treat the code as a sketch to
> check against the library's current README, not as a verified integration.

## Model contract

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

## Sketch

```ts
import { loadTensorflowModel } from 'react-native-fast-tflite';

const model = await loadTensorflowModel(require('./assets/nutrition.tflite'), 'android-gpu');
// rgb: Uint8Array of length 224 * 224 * 3, RGB interleaved (R,G,B,R,G,B,...)
const [out] = await model.run([rgb]);
const [calories, mass, fat, carb, protein] = new Float32Array(out);
```

Delegates documented by the library: `'android-gpu'`, `'nnapi'` (deprecated on
Android 15 per its README); omit the argument for plain CPU. Check the README for
the exact input/output typing before relying on the sketch above.

## Open spikes (not yet researched)

1. **Pixels from a photo.** Getting a still photo (camera or gallery) resized to
   224x224 and into a raw RGB byte array inside Expo has not been investigated.
   Candidates to evaluate: VisionCamera frame processors, or decoding a resized
   image in JS or native code.
2. **Resize parity.** Training uses PIL bilinear resizing. Run the same photos
   through `python -m calorie_model.predict` and through the app; if outputs
   differ noticeably, either match the resampler in training or add
   resize-method augmentation.
3. **Latency and accuracy on real hardware**, CPU vs GPU delegate, fp32 vs int8.
