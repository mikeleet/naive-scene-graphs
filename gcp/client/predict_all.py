import requests, json, numpy as np, time
from sklearn.naive_bayes import CategoricalNB
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

NB_URL = "URL"
VISUAL_PRETRAINED_URL = "URL"
VISUAL_TRAINED_URL = "URL"

def call_nb(pairs, img_w=1024, img_h=1024):
    r = requests.post(f"{NB_URL}/predict", json={
        "image_width": img_w, "image_height": img_h, "pairs": pairs
    })
    return r.json()["predictions"] if r.status_code == 200 else []

def call_visual(url, boxes, image_url=""):
    r = requests.post(f"{url}/embed", json={
        "image_url": image_url, "boxes": boxes
    })
    return r.json()["embeddings"] if r.status_code == 200 else []

def discretize(embeddings, n_components=8, n_clusters=16):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(np.array(embeddings))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    bins = kmeans.fit_predict(reduced)
    return bins

print("testing with sample pairs...")
test_pairs = [
    {"head_class": "person", "tail_class": "horse",
     "head_x1": 100, "head_y1": 50, "head_x2": 300, "head_y2": 400,
     "tail_x1": 250, "tail_y1": 100, "tail_x2": 800, "tail_y2": 600},
]

nb_preds = call_nb(test_pairs)
print("geometric (NB):")
for p in nb_preds[:2]:
    print(f"  {p['head']} --{p['predicate']}--> {p['tail']} (conf={p['confidence']:.3f})")

boxes = [{"x1": p["head_x1"], "y1": p["head_y1"], "x2": p["head_x2"], "y2": p["head_y2"]} for p in test_pairs]
emb_p = call_visual(VISUAL_PRETRAINED_URL, boxes)
emb_t = call_visual(VISUAL_TRAINED_URL, boxes) if VISUAL_TRAINED_URL != "URL" else []

print(f"\npretrained embedding dim: {len(emb_p[0]) if emb_p else 0}")
print(f"trained embedding dim: {len(emb_t[0]) if emb_t else 0}")

if emb_p and emb_t:
    print("\ndiscretized features (8 components, 16 clusters):")
    bins_p = discretize(emb_p)
    bins_t = discretize(emb_t)
    print(f"  pretrained: {bins_p}")
    print(f"  trained: {bins_t}")

print("\nnote: full comparison requires both endpoints deployed.")
print("set NB_URL, VISUAL_PRETRAINED_URL, VISUAL_TRAINED_URL above.")
