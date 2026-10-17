#!/usr/bin/env bash
# Download the parts of Nutrition5k needed for RGB training (overhead images,
# labels, splits). Requires the Google Cloud SDK (gsutil). The full dataset is
# large; this skips the depth images and side-angle videos.
set -euo pipefail

DEST="${1:-data/nutrition5k}"
SRC="gs://nutrition5k_dataset/nutrition5k_dataset"

mkdir -p "$DEST/metadata" "$DEST/dish_ids" "$DEST/imagery"
gsutil -m cp "$SRC/metadata/dish_metadata_cafe*.csv" "$DEST/metadata/"
gsutil -m cp -r "$SRC/dish_ids/splits" "$DEST/dish_ids/"
gsutil -m rsync -r -x '.*\.(h5|avi|mp4)$|depth.*' "$SRC/imagery/realsense_overhead" "$DEST/imagery/realsense_overhead"
echo "Downloaded to $DEST — check the layout matches the docstring in src/calorie_model/data.py"
