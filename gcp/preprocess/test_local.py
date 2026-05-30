import os, json, zipfile, io, random
from pathlib import Path
from PIL import Image as PILImage

PROJECT = Path("/Users/mike/Documents/GitHub/non_sn/app-11-dl-on-gcp/naive-scene-graphs")
DATA = PROJECT / "data"
IMAGES = Path("/Users/mike/Documents/GitHub/non_sn/app-11-dl-on-gcp/img")
OUTPUT = Path(__file__).resolve().parent / "output"
OUTPUT.mkdir(exist_ok=True)

IMG_SIZE = 224
random.seed(42)

# load dicts + v1.2 data
with open(DATA / "vg150/VG-SGG-dicts.json") as f:
    d = json.load(f)
label_to_idx = d["label_to_idx"]  # {"airplane": "1", "animal": "2", ...}
idx_to_label = {int(k)-1: v for k, v in d["idx_to_label"].items()}

with open(DATA / "vg150/annotations/image_data.json") as f:
    img_data = json.load(f)
id_to_url = {item["image_id"]: item["url"] for item in img_data}
id_to_folder = {vid: url.split("/")[-2] for vid, url in id_to_url.items()}

with zipfile.ZipFile(DATA / "vg150/annotations/relationships_v1_2.json.zip", "r") as zf:
    with zf.open("relationships.json") as f:
        v12 = json.load(f)

print(f"v1.2 entries: {len(v12)}, VG150 classes: {len(label_to_idx)}")

# open zips
zip1 = zipfile.ZipFile(IMAGES / "images.zip", "r")
zip2 = zipfile.ZipFile(IMAGES / "images2.zip", "r")
zips = {"VG_100K": zip1, "VG_100K_2": zip2}

# pick random images and crop their boxes using v1.2 coords
test_entries = random.sample(range(len(v12)), 10)
crops_done = 0

print(f"\ntesting on {len(test_entries)} random images...\n")

for entry_idx in test_entries:
    entry = v12[entry_idx]
    vg_id = entry["image_id"]
    folder = id_to_folder.get(vg_id, "VG_100K")
    zf = zips.get(folder)
    if zf is None:
        print(f"  VG {vg_id}: zip not found")
        continue

    img_name = f"{folder}/{vg_id}.jpg"
    if img_name not in zf.namelist():
        print(f"  VG {vg_id}: not in zip")
        continue

    try:
        img_bytes = zf.read(img_name)
        img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        print(f"  VG {vg_id}: corrupt")
        continue

    img_w, img_h = img.size

    # collect unique objects from v1.2 relationships
    boxes = {}
    for rel in entry.get("relationships", []):
        for side in ("subject", "object"):
            obj = rel.get(side, {})
            oid = obj.get("object_id")
            if oid and oid not in boxes:
                x = obj.get("x", 0)
                y = obj.get("y", 0)
                w = obj.get("w", 0)
                h = obj.get("h", 0)
                name = obj.get("name", "?")
                if isinstance(name, list):
                    name = name[0] if name else "?"
                name = str(name)

                # vg150 class index (map raw name -> class index)
                cls_idx = int(label_to_idx.get(name, -1))
                if cls_idx < 0:
                    continue

                boxes[oid] = (x, y, w, h, name, cls_idx - 1)

    if not boxes:
        continue

    print(f"  VG {vg_id}: {img_w}x{img_h}, {len(boxes)} VG150 objects")

    sample_boxes = random.sample(list(boxes.values()), min(3, len(boxes)))
    for x, y, w, h, name, cls_idx in sample_boxes:
        if w <= 0 or h <= 0:
            continue
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = img.crop((x1, y1, x2, y2))
        crop = crop.resize((IMG_SIZE, IMG_SIZE), PILImage.BILINEAR)

        vg150_name = idx_to_label.get(cls_idx, f"id_{cls_idx}")
        out_name = f"crop_{vg_id}_{name}_{vg150_name}.jpg"
        crop.save(OUTPUT / out_name, "JPEG", quality=85)
        print(f"    {name:15s} ({vg150_name:12s}) {w:3d}x{h:3d} → {out_name}")
        crops_done += 1

zip1.close()
zip2.close()

print(f"\n{crops_done} crops saved to {OUTPUT}/")

crop_files = sorted(OUTPUT.glob("crop_*.jpg"))
print(f"\nall crops ({len(crop_files)}):")
for f in crop_files:
    print(f"  {f.name}")
