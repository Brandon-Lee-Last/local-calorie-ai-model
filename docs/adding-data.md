# Adding more training data

The model learns calories and macros from photos with nutrition labels. Nutrition5k is the only
big, trustworthy source wired in so far (5,006 dishes, weighed). Halving its dishes made calorie error
about 16% worse, so more *good* labelled data should help; the catch is that there is not a lot of it.
This page is how to plug in any source you find, and what I found.

## What a source needs

Photos plus **any** of: calories (kcal), mass (g), fat (g), carbs (g), protein (g). Labels may be partial:
a source with only weights trains only the weight output, one with only calories trains only calories.
A source with no nutrition labels at all (only "this is food") still helps the food / not-food output via
`--openimages-root`.

## The format

A folder with an `index.csv` (and the photos, any size or format PIL can read; they are squashed to the
model size when loaded). Only `path` is required:

| column | meaning |
|---|---|
| `path` | photo, relative to the folder |
| `calories, mass, fat, carb, protein` | real units; **leave blank when unknown** |
| `weight` | how far to trust this photo's labels, default 1. Use about 1 for lab-measured labels and 0.3-0.5 for estimated or AI-generated ones, so they steer the model without dominating |
| `split` | `test` keeps the photo out of training and reports it separately; anything else is training |
| `dish_id`, `source` | informational |

## Steps

1. Convert the dataset into that folder with a small script. Template:
   ```python
   import csv, pathlib
   out = pathlib.Path("data/fpb")                      # photos go under data/fpb/img/...
   with open(out / "index.csv", "w", newline="") as f:
       w = csv.DictWriter(f, ["path", "mass", "weight", "split"])
       w.writeheader()
       for photo, grams, is_test in read_their_labels():   # <- their format
           w.writerow({"path": f"img/{photo}", "mass": grams, "weight": 1, "split": "test" if is_test else ""})
   ```
2. Train with it (repeat `--extra-data` for several sources):
   ```bash
   python -m calorie_model.train --openimages-root data/openimages --extra-data data/fpb --out runs/with_fpb
   ```
3. **Judge it on Nutrition5k, not on itself.** Compare the TEST table's `all` and `view_overhead` rows with
   a previous run. Keep the source only if calorie error on the official test split improves by clearly
   more than the ~2 kcal run-to-run noise. The report also lists the source's own held-out photos as
   `extra_<name>`. The checkpoint is still chosen on clean Nutrition5k validation, so noisy sources
   cannot steer the choice.
4. Tune `weight`. Noisy labels at weight 1 can hurt.

## Candidates

**None of these was downloaded or licence-checked by me.** Their hosts (Kaggle, Hugging Face, ISSAI,
GitHub repos outside this project) were blocked from my environment, so everything below is from search
summaries and papers. Open each page and read the licence and the label provenance first.

| Source | What it is | Caveats |
|---|---|---|
| Food Portion Benchmark (FPB), ISSAI / Nazarbayev University | 14,083 photos, 138 classes of Central Asian food, lab-measured component weights, bounding boxes, several angles per dish ([page](https://issai.nu.edu.kz/fpb/)) | Weights, not calories (energy would have to be derived from a nutrient table); licence not seen |
| MM-Food-100K (Codatta) | 100,000 photos with calories and estimated ingredient weights ([Hugging Face](https://huggingface.co/datasets/Codatta-dev/MM-Food), [paper](https://arxiv.org/pdf/2508.10429)) | Labels produced by GPT-4o / Qwen-max with human review, so **estimates, not measurements**: use a low `weight`; no licence text found; the owners keep ~90% of the corpus for commercial use |
| NutritionVerse-Synth | ~85,000 synthetic images from 7,082 generated scenes with mass, calories and macros ([paper](https://arxiv.org/pdf/2312.06192)) | Rendered, not photographed, so a domain gap; on Kaggle; no licence found |
| ECUSTFD | 2,978 photos, 19 food types, mass and volume, a coin for scale ([paper](https://arxiv.org/pdf/1705.07632)) | Small, plain backgrounds, Chinese foods |
| SimpleFood45, ACETADA, FinnFoodPics | 513 / 806 / 72 photos with energy labels | Small; useful for testing more than training |

## Before using any source

- Does the licence allow commercial use, and what attribution does it require?
- Are the labels measured (weighed) or estimated? That sets the `weight`.
- Do its photos duplicate another source?
- Whether training on a dataset needs more than attribution is a question for someone qualified.

## What would help most

(For South African food, `docs/south-african-foods.md` turns this into a procedure and a converter: `ingest_meals`.)

A few hundred **phone photos of meals you weighed**, with the real weight (and calories if you know them).
That is the exact domain the app works in, no public dataset has it at scale, and even 200-500 photos would
be worth more than a large pile of estimated labels, both for fine-tuning and for knowing how good the app
really is. Put them in this format with `weight` 1 and `split` = `test` for about a third of them.
