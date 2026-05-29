FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir scikit-learn numpy shapely h5py

COPY trf_baseline/ /app/trf_baseline/
COPY implementations/ /app/implementations/
ENV PYTHONPATH=/app

CMD ["python", "-m", "trf_baseline.preprocess"]
