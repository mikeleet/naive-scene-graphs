FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir Pillow h5py google-cloud-storage numpy psutil

COPY preprocess.py /app/preprocess.py

ENTRYPOINT ["python", "preprocess.py"]
