#!/bin/bash

cd "$(dirname "$0")/../.."
set -e

REGION="us-central1"
PROJECT=$(gcloud config get-value project)
REPO="$REGION-docker.pkg.dev/$PROJECT/scene-graph"

echo "=== deploy NB (linux/amd64 — cloud run is x86) ==="
docker build --platform linux/amd64 -t scene-graph-nb -f deploy/Dockerfile deploy/
docker tag scene-graph-nb $REPO/scene-graph-nb:latest
docker push $REPO/scene-graph-nb:latest
echo "done"
