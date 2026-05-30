import json, os, random
from pathlib import Path
from PIL import Image

INDEX_PATH = "/gcs/crops/index.jsonl" if os.path.exists("/gcs") else "/tmp/crops/index.jsonl"
CROPS_DIR  = "/gcs/crops" if os.path.exists("/gcs") else "/tmp/crops"

if not os.path.exists(INDEX_PATH):
    print(f"FAIL: index not found at {INDEX_PATH}")
    exit(1)

entries = []
with open(INDEX_PATH) as f:
    for line in f:
        entries.append(json.loads(line))

print(f"index: {len(entries)} entries")

train = [e for e in entries if e["split"] == 0]
test  = [e for e in entries if e["split"] == 2]
print(f"train: {len(train)}, test: {len(test)}, ratio: {len(train)/(len(train)+len(test)):.1%}")

labels = set(e["label"] for e in entries)
print(f"unique labels: {len(labels)}")

class_counts = {}
for e in entries:
    class_counts[e["label"]] = class_counts.get(e["label"], 0) + 1
top5 = sorted(class_counts.items(), key=lambda x: -x[1])[:5]
print(f"top 5 classes: {top5}")
print(f"classes with 0: {sum(1 for k in range(150) if k not in class_counts)}")

sample = random.sample(entries, min(100, len(entries)))
bad = 0
for e in sample:
    path = f"{CROPS_DIR}/{e['crop']}"
    try:
        img = Image.open(path)
        w, h = img.size
        if w != 224 or h != 224:
            print(f"  wrong size: {e['crop']} {w}x{h}")
            bad += 1
    except Exception:
        print(f"  corrupt: {e['crop']}")
        bad += 1

print(f"bad crops: {bad}/{len(sample)}")

with open("class_weights.json") as f:
    w = json.load(f)
print(f"class weights: {len(w)} entries")
print(f"  min: {min(w.values()):.2f}, max: {max(w.values()):.2f}")

print("\nALL CHECKS PASSED" if bad == 0 and len(labels) == 150 else "\nISSUES FOUND")
