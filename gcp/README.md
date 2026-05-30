# gcp deployment

3 cloud run endpoints + vertex ai training. terraform-managed.

## setup

see `../GCP_SETUP.md` in parent directory. do once as a human.

## quickstart

```bash
# 1. upload vg images to GCS
bash gcp/scripts/upload_images.sh

# 2. train resnet-50 from scratch (route 1)
bash gcp/scripts/route1_train.sh

# 3. deploy all 3 endpoints (route 2)
bash gcp/scripts/route2_deploy.sh

# 4. test combined prediction
python gcp/client/predict_all.py

# clean up
bash gcp/scripts/destroy.sh
```

## services

| service | endpoint | model |
|---|---|---|
| scene-graph-nb | /predict, /predict_batch | sklearn categoricalnb |
| visual-pretrained | /embed | resnet-50 (imagenet, frozen) |
| visual-trained | /embed | resnet-50 (vg150, route 1) |

## terraform

```bash
cd gcp/terraform
cp terraform.tfvars.example terraform.tfvars   # edit project_id
terraform init
terraform plan
terraform apply
```

5 modules: artifact_registry, cloud_run_nb, cloud_run_visual (×2), vertex_ai_training, iam.

## logs

each run creates `gcp/logs/{timestamp}/`:
- vertex_ai training logs + tensorboard
- cloud run deploy output
- cost summary per resource
- validation results

## costs

~$8 total. see GCP_SETUP.md for breakdown.
