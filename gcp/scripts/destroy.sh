#!/bin/bash
set -e

PROJECT=$(gcloud config get-value project)
echo "terraform destroy for project: $PROJECT"
read -p "confirm? [y/N] " confirm
[ "$confirm" != "y" ] && exit 0

terraform -chdir=gcp/terraform destroy -auto-approve

echo "done. check console for any remainders:"
echo "  gcloud run services list"
echo "  gcloud ai custom-jobs list"
echo "  gsutil ls"
