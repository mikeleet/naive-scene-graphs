#!/bin/bash
set -e

if [ -n "$CHECKPOINT_GCS" ]; then
    echo "loading checkpoint from $CHECKPOINT_GCS"
    gsutil cp "$CHECKPOINT_GCS" /app/best_checkpoint.pth
fi

exec uvicorn serve:app --host 0.0.0.0 --port 8080
