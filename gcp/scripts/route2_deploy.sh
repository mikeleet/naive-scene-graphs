#!/bin/bash

cd "$(dirname "$0")/../.."
set -e

REGION="us-central1"
PROJECT=$(gcloud config get-value project)
REPO="$REGION-docker.pkg.dev/$PROJECT/scene-graph"

echo "=== build visual-pretrained (linux/amd64 — cloud run is x86) ==="
docker build --platform linux/amd64 -t visual-pretrained -f gcp/route2_visual/Dockerfile.pretrained gcp/route2_visual/

echo "=== build visual-trained (linux/amd64 — cloud run is x86) ==="
docker build --platform linux/amd64 -t visual-trained -f gcp/route2_visual/Dockerfile.trained gcp/route2_visual/
docker tag visual-trained $REPO/visual-trained:latest
docker push $REPO/visual-trained:latest

echo "=== deploy ==="
terraform -chdir=gcp/terraform init
terraform -chdir=gcp/terraform plan
terraform -chdir=gcp/terraform apply -auto-approve

echo ""
echo "endpoints:"
echo "  NB:      $(terraform -chdir=gcp/terraform output -raw nb_service_url)/health"
echo "  VisualP: $(terraform -chdir=gcp/terraform output -raw visual_pretrained_url)/health"
echo "  VisualT: $(terraform -chdir=gcp/terraform output -raw visual_trained_url)/health"
