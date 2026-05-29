FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir scikit-learn numpy fastapi uvicorn shapely

COPY model/ /app/model/
COPY serve.py /app/serve.py

EXPOSE 9500
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "9500"]
