#!/bin/bash
set -e

BUCKET="gs://metal-ranger-123605-scene-graph"
PROJECT_ROOT="/Users/mike/Documents/GitHub/non_sn/app-11-dl-on-gcp/naive-scene-graphs"

echo "uploading to $BUCKET"

echo "--- images ---"
gsutil cp /Users/mike/Documents/GitHub/non_sn/app-11-dl-on-gcp/img/images.zip   $BUCKET/vg/images.zip
gsutil cp /Users/mike/Documents/GitHub/non_sn/app-11-dl-on-gcp/img/images2.zip  $BUCKET/vg/images2.zip

echo "--- h5 + mapping ---"
gsutil cp $PROJECT_ROOT/data/vg150/VG-SGG.h5              $BUCKET/vg/VG-SGG.h5
gsutil cp $PROJECT_ROOT/data/vg150/VG-SGG-dicts.json      $BUCKET/vg/VG-SGG-dicts.json
gsutil cp $PROJECT_ROOT/data/vg150/h5_to_vg_mapping.json  $BUCKET/vg/h5_to_vg_mapping.json
gsutil cp $PROJECT_ROOT/data/vg150/annotations/image_data.json $BUCKET/vg/image_data.json

echo "--- relationships ---"
gsutil cp $PROJECT_ROOT/data/vg150/annotations/relationships_v1_2.json.zip $BUCKET/vg/relationships_v1_2.json.zip

echo "done"
gsutil ls $BUCKET/vg/
