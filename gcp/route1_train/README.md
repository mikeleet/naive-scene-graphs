# route 1 — train resnet-50 from scratch on vg150 crops

usage:
```bash
bash gcp/scripts/route1_train.sh
```

## what it does

1. builds docker image with pytorch + torchvision
2. pushes to artifact registry
3. submits vertex ai custom training job (t4 gpu)
4. downloads logs + tensorboard + checkpoint after completion

## training config

- resnet-50, random init, 150 classes
- adam, lr=0.001, batch_size=64, 10 epochs
- crops: bounding boxes resized to 224x224
- output: best_checkpoint.pth → gcs

## vertex ai job

```bash
gcloud beta ai custom-jobs create \
  --region=us-central1 \
  --display-name=resnet50-vg150 \
  --config=gcp/route1_train/vertex_config.yaml
```

## logs

training output logged to `gcp/logs/{timestamp}_train/`:
- hyperparams.json
- metrics_per_epoch.csv
- tensorboard event files (downloaded)
- training_log.txt (full stdout)

## costs

t4 gpu: ~2 hours × $0.60/hr = **~$1.20**
