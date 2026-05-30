FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir torchvision google-cloud-storage

COPY train.py /app/train.py

ENTRYPOINT ["python", "train.py"]
