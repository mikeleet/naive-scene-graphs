#!/bin/bash

cd "$(dirname "$0")/../.."
set -e

REGION="us-central1"
PROJECT=$(gcloud config get-value project)
IMAGE="resnet50-train"
REPO="$REGION-docker.pkg.dev/$PROJECT/scene-graph"

echo "=== build (linux/amd64 — vertex ai is x86 not arm) ==="
docker build --platform linux/amd64 -t $IMAGE -f gcp/route1_train/Dockerfile gcp/route1_train/

echo "=== tag + push ==="
docker tag $IMAGE $REPO/$IMAGE:latest
docker push $REPO/$IMAGE:latest

echo "=== submit job ==="
gcloud beta ai custom-jobs create \
  --region=$REGION \
  --display-name=resnet50-vg150 \
  --config=gcp/route1_train/vertex_config.yaml
