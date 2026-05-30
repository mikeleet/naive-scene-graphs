import os, json, zipfile, io, time, tarfile, threading, sys, traceback
sys.stdout.reconfigure(line_buffering=True)
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image as PILImage
from google.cloud import storage
from collections import Counter
import psutil

BUCKET = os.getenv("BUCKET", "metal-ranger-123605-scene-graph")
TMP = "/tmp"
CROPS_PER_TAR = 10000
IMG_SIZE = 224
NUM_WORKERS = 4
UPLOAD_RETRIES = 3

os.makedirs(TMP, exist_ok=True)
client = storage.Client()
bucket = client.bucket(BUCKET)

def gcs_write(path, data):
    bucket.blob(path).upload_from_string(data)

def upload_file(local, remote, retries=UPLOAD_RETRIES):
    for attempt in range(retries):
        try:
            bucket.blob(remote).upload_from_filename(local)
            os.remove(local)
            return True
        except Exception as e:
            if attempt == retries - 1:
                print(f"    upload FAILED after {retries} retries: {remote} ({e})")
                return False
            time.sleep(2 ** attempt)

print(f"bucket: {BUCKET}, workers: {NUM_WORKERS}, crops/tar: {CROPS_PER_TAR}")

if bucket.blob("crops/_SUCCESS").exists():
    print("_SUCCESS marker found — preprocessing already complete, exiting")
    exit(0)

print("downloading zips to local ssd...")
for name in ["images.zip", "images2.zip", "VG-SGG-dicts.json",
             "image_data.json", "relationships_v1_2.json.zip", "VG-SGG.h5"]:
    local_path = f"{TMP}/{name}"
    if os.path.exists(local_path):
        continue
    blob = bucket.blob(f"vg/{name}")
    blob.download_to_filename(local_path)
    print(f"  {name}: {os.path.getsize(local_path)/1024**2:.0f} MB")

print("loading dicts + split data...")
with open(f"{TMP}/VG-SGG-dicts.json") as f:
    label_to_idx = json.load(f)["label_to_idx"]

with open(f"{TMP}/image_data.json") as f:
    img_data = json.load(f)
id_to_folder = {item["image_id"]: item["url"].split("/")[-2] for item in img_data}

with zipfile.ZipFile(f"{TMP}/relationships_v1_2.json.zip", "r") as zf:
    v12 = json.loads(zf.read("relationships.json"))

import h5py
with h5py.File(f"{TMP}/VG-SGG.h5", "r") as f:
    h5_splits = f["split"][:]

print(f"v1.2: {len(v12)} images, vg150: {len(label_to_idx)} classes")
print(f"cores: {psutil.cpu_count(logical=True)} logical, {psutil.cpu_count(logical=False)} physical")

zip1 = zipfile.ZipFile(f"{TMP}/images.zip", "r")
zip2 = zipfile.ZipFile(f"{TMP}/images2.zip", "r")
zips = {"VG_100K": zip1, "VG_100K_2": zip2}

# per-worker state
workers = [{"tar": None, "count": 0, "index": 0, "entries": []}
           for _ in range(NUM_WORKERS)]
upload_threads = []

def close_and_upload(w, tar_name):
    if w["tar"] is None:
        return
    w["tar"].close()
    local = f"{TMP}/{tar_name}"
    t = threading.Thread(target=upload_file, args=(local, f"crops/{tar_name}"))
    t.start()
    upload_threads.append(t)
    print(f"    uploaded {tar_name} ({w['count']} crops)")
    w["tar"] = None
    w["count"] = 0

def new_tar(w, prefix, worker_id):
    w["index"] += 1
    name = f"{prefix}_w{worker_id}_{w['index']:04d}.tar"
    w["tar"] = tarfile.open(f"{TMP}/{name}", "w")
    w["count"] = 0
    return name

# process a batch of crops (given as (jpeg_bytes, name, label, is_train))
def process_batch(crops_batch):
    results = []
    for name, jpeg_bytes, box, label, is_train in crops_batch:
        img = PILImage.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        x, y, w, h = box
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(img.width, int(x+w)), min(img.height, int(y+h))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img.crop((x1, y1, x2, y2))
        crop = crop.resize((IMG_SIZE, IMG_SIZE), PILImage.BILINEAR)
        buf = io.BytesIO()
        crop.save(buf, "JPEG", quality=85)
        results.append((name, buf.getvalue(), label, is_train))
    return results
