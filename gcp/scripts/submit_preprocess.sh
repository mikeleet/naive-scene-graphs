#!/bin/bash
set -e

cd "$(dirname "$0")/../.."

REGION="us-central1"
PROJECT="metal-ranger-123605"
REPO="$REGION-docker.pkg.dev/$PROJECT/scene-graph"

echo "=== cancel previous job (if any) ==="
gcloud beta ai custom-jobs cancel projects/1008176684587/locations/us-central1/customJobs/5508452632663949312 2>/dev/null || true

echo "=== build (linux/amd64 — vertex ai is x86 not arm) ==="
docker build --platform linux/amd64 -t preprocess -f gcp/preprocess/Dockerfile gcp/preprocess/

echo "=== push ==="
docker tag preprocess $REPO/preprocess:latest
docker push $REPO/preprocess:latest

echo "=== submit preprocess job ==="
gcloud beta ai custom-jobs create \
  --region=$REGION \
  --display-name=vg150-preprocess \
  --config=gcp/preprocess/vertex_config.yaml

echo "done. check status: gcloud beta ai custom-jobs list"
