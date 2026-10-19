# South African food

The model does not know South African food yet. This page says exactly where that stands, what can honestly
be claimed about it, and the steps that change it.

## Where it stands today

- **v0 has never seen a South African dish.** It learned from Nutrition5k (US cafeteria plates) and, for the
  food / not-food output, generic Open Images photos. Ask it about bunny chow and it will guess from whatever
  looks closest. Nothing in this repo measures how wrong it is on South African food, and I have not checked
  whether the food / not-food output would reject pap or boerewors as "not food".
- **I could not find a public dataset of South African food photos with weights or nutrition.** What exists is
  small and from elsewhere in Africa (a six-food Ghana/Cameroon set, a 1,658-image six-category set, an
  11-dish Ivorian set that was not released). My environment cannot reach Wikimedia Commons or Flickr, so a
  search through those has to run on your PC (below).
- **The SAFOODS composition tables (SAMRC) are the natural source of nutrient values**, but I found no licence
  terms, the last update I saw was 2017, and reports say only about a third of its values are analysed rather
  than borrowed. I have not copied anything from it, and nothing in `nutrition_per_100g.csv` comes from memory.

## What can be said, and when

"The first South African AI model that knows our food" cannot be stated, and I would not use it:

- Nobody can verify "first". A calorie app called **CalZA** lists itself on the app stores as "South Africa's
  first AI-powered calorie tracker built specifically for SA foods" (233 local dishes) according to a search
  result I could not open to check. That is its developer's own claim, and I have no data on how accurate it
  is, but it means "first" is already contested.
- "Knows" has to be a number. A claim you can defend is one you measured. Use this ladder and stop at the
  highest rung you have evidence for:

| Evidence you have | What you can say |
|---|---|
| Nothing yet (today) | "Runs on your phone, nothing is uploaded. Being built around South African food." |
| Commons photos + a dish recognition head, tested on held-out photos | "Recognises N South African dishes (top-1 X% on photos it had not seen)." |
| Weighed meals, run through the held-out test below | "On N weighed South African meals it is typically within X kcal, against Y kcal for the model without South African training and Z kcal for guessing the same number every time." Publish the dishes it is bad at too. |

A claim like the last one is stronger than "first": it can be checked, and it is true of the model you ship.

## The plan

Three kinds of data, doing three different jobs:

| Data | Gives | Cannot give |
|---|---|---|
| Openly licensed photos (Wikimedia Commons) | The food / not-food output learns that South African food is food; a pool of examples per dish for a dish recogniser | Any calories, mass or macros |
| **Weighed phone photos of real meals** | Mass in grams (straight from the scale), and calories/macros through the nutrient sheet; the only thing that can say how good the app is on this food | Volume by itself: a few hundred meals is a start, not coverage |
| Per-100 g nutrient sheet | Turns "250 g of pap and 150 g of boerewors" into calories and macros | Anything without a sourced number: it ships empty |

### 1. The dish list

`datasets/south_african/dishes.csv`: 57 dishes with aliases, a category, a one-line description and search
terms. It was written from general knowledge, so **treat it as a first draft to correct**: delete dishes you
would not recognise on a plate, fix names and aliases, add what is missing (columns `dish_id`, `name`, `aliases` separated by `|`,
`category`, `description`, `commons_queries` separated by `;`; add the same `dish_id` as a new row at the end of
`nutrition_per_100g.csv`). Dishes that are not specifically South African (meat pies, burgers) are mostly not
listed, and plates are described by their parts (pap + boerewors + chakalaka).

### 2. Openly licensed photos (run on your PC)

```powershell
python -m calorie_model.prepare_commons download --out data/sa_commons --contact you@example.com --dish bunny_chow --per-dish 5
```

**This downloader has been tested only against made-up responses shaped like the Commons API, not the live
site.** Run it on one dish first, and tell me what it prints if anything looks off. Then all dishes (about
40 per dish, ~1 second per request, so expect an hour or so):

```powershell
python -m calorie_model.prepare_commons download --out data/sa_commons --contact you@example.com
```

Open `data/sa_commons/img/<dish>/` and **delete every photo that is not that dish** (Commons search is by text;
expect menus, maps, wrong dishes). Deleted photos are remembered and not downloaded again. Then:

```powershell
python -m calorie_model.prepare_commons index --out data/sa_commons
```

What it keeps: public domain, CC0 and CC BY photos, with the author, licence and page link in
`attribution.csv`. CC BY needs the author credited, so keep that file and show credits in the app. CC BY-SA is
left out unless you pass `--allow-sharealike`; whether share-alike reaches a trained model's weights is
unsettled, so ask someone qualified before turning it on. NonCommercial and NoDerivatives photos are never kept.
A photographer's pictures are all on the same side of the train/test split.

### 3. Weighed meals (this is the part that matters)

You need a kitchen scale (1 g steps), the phone you will use, and plates of real South African food: home
cooking, takeaways, braais, street food. For each meal:

1. Put the plate on the scale and tare it. Add one component at a time and write down the grams for each
   (`pap` 250, `boerewors` 150, `chakalaka` 80). Weigh what is on the plate, bones and shells included only if
   you will eat them.
2. Photograph it before eating, two or three photos from different angles (overhead and about 45 degrees).
   Vary plates, tables, light and phones. Do not tidy the plate for the camera.
3. One row in `meals.csv` per *meal* (see `datasets/south_african/meals_template.csv`):

   ```
   meal_id,photos,components,calories,fat,carb,protein
   m001,IMG_0001.jpg|IMG_0002.jpg,pap:250|boerewors:150|chakalaka:80,,,,
   ```

   Dish ids, names and aliases all work; typos get a "did you mean". `calories`/`fat`/`carb`/`protein` are optional
   whole-meal totals if you really have them (a label, a lab). Leave them blank otherwise.
4. Convert:

   ```powershell
   python -m calorie_model.ingest_meals --meals meals.csv --photos-dir phone_photos --out data/sa_meals
   ```

   Photos are made upright, shrunk and stripped of metadata (GPS position, camera serial). A random 30% of
   *meals* (not photos) is held out as the test set; the same meal always lands on the same side.

How many: with the Nutrition5k learning curve (each doubling of dishes cut calorie error by about 14%) in mind,
**200-500 meals is the point at which this starts to show; under about 10 meals of a dish tells you nothing about
that dish.** A held-out test set worth quoting needs at least about 100 meals across at least 20 dishes.

If you collect other people's meals, ask for consent first, and think about POPIA (South Africa's data
protection law) before building an in-app contribution feature: what you store, how a person deletes theirs.
I am not qualified to advise on that.

### 4. The nutrient sheet

`datasets/south_african/nutrition_per_100g.csv` has one blank row per dish. Fill a row only from a source you can
name, and write it in the `source` and `source_ref` columns: a SAFOODS table (after confirming you may use it),
the nutrition label of a packaged product (biltong, rusks), a lab result, or a dietitian. The loader refuses a
row with numbers and no source, and warns when a row's calories disagree with its own fat/carb/protein by more
than 25% (4 kcal/g protein and carbs, 9 kcal/g fat), which usually means a typo.

A row is a single average for a dish that really varies with recipe, oil and filling. That is why meals whose
nutrients come from the sheet get a lower training weight (`--table-weight`, default 0.5), and why mass, which
is weighed, is the better-founded output. Until the sheet has values, weighed meals train mass only.

### 5. Train and test

```powershell
# train with South African data
python -m calorie_model.train --openimages-root data/openimages --extra-data data/sa_commons --extra-data data/sa_meals --out runs/sa

# the same held-out meals, scored on a run that never saw South African food
python -m calorie_model.train --openimages-root data/openimages --holdout-data data/sa_meals --out runs/no_sa
```

Both runs print, for `sa_meals`, the error on the held-out meals (`extra_sa_meals`) and for always guessing the
Nutrition5k median (`extra_sa_meals_guess`). For `sa_commons` the report shows `food_recall`: the share of
South African food photos accepted as food. Those three comparisons (with SA data, without, guessing) are the
evidence for the claim above. `runs/sa` should also still do as well as before on Nutrition5k, or something is
wrong.

## What is not built, and why

- **A dish recognition head** ("this is bunny chow, 94%"). It needs the photos from step 2 and the answer to
  whether a few dozen photos per dish is enough, which I can measure once they exist. Building it first would mean
  shipping a dish name with nothing to show it is right.
- **An in-app "log what you actually ate and weighed" flow** to collect more meals from users. Worth doing, but
  it needs consent and deletion decided first (above).
- **App copy.** The app says nothing about South African food, and should not until the numbers exist.

## What I need from you

1. Correct `dishes.csv`.
2. Run the Commons downloader on one dish, then all, and review the photos.
3. Weigh and photograph meals. Even 30 meals is enough to find out whether the whole path works.
4. Decide where nutrient values will come from, and check the licence if it is SAFOODS.
