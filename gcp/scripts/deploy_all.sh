#!/bin/bash

cd "$(dirname "$0")/../.."
set -e

REGION="us-central1"
PROJECT=$(gcloud config get-value project)
REPO="$REGION-docker.pkg.dev/$PROJECT/scene-graph"

echo "=== deploy nb ==="
docker build -t scene-graph-nb -f deploy/Dockerfile deploy/
docker tag scene-graph-nb $REPO/scene-graph-nb:latest
docker push $REPO/scene-graph-nb:latest

echo "=== terraform apply ==="
terraform -chdir=gcp/terraform init
terraform -chdir=gcp/terraform plan
terraform -chdir=gcp/terraform apply -auto-approve

echo "=== deploy visual-pretrained ==="
docker build -t visual-pretrained -f gcp/route2_visual/Dockerfile.pretrained gcp/route2_visual/
docker tag visual-pretrained $REPO/visual-pretrained:latest
docker push $REPO/visual-pretrained:latest

echo "=== deploy visual-trained ==="
docker build -t visual-trained -f gcp/route2_visual/Dockerfile.trained gcp/route2_visual/
docker tag visual-trained $REPO/visual-trained:latest
docker push $REPO/visual-trained:latest

echo "done"
