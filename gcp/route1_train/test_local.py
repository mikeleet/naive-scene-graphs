import os, json, h5py, zipfile, io, random
from pathlib import Path
from PIL import Image as PILImage
import torch, torchvision
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT = Path("/Users/mike/Documents/GitHub/non_sn/app-11-dl-on-gcp/naive-scene-graphs/gcp/preprocess/output")

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
])

print(f"device: {DEVICE}")

crop_files = sorted(OUTPUT.glob("crop_*.jpg"))
if not crop_files:
    print("no crops found. run test_local.py first.")
    exit(1)

print(f"loading pretrained resnet-50...")
model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
model = torch.nn.Sequential(*list(model.children())[:-1])
model = model.to(DEVICE).eval()

print(f"testing on {len(crop_files)} crops\n")

embeddings = []
for path in crop_files[:5]:
    img = PILImage.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model(tensor).squeeze().cpu().numpy()

    label = path.stem.split("_")[-1]
    embeddings.append(emb)
    print(f"  {path.name:45s} dim={emb.shape} mean={emb.mean():.4f} "
          f"std={emb.std():.4f} label={label}")

embeddings = np.array(embeddings)
print(f"\nembeddings: {len(embeddings)} × {embeddings.shape[1]}")
print(f"pca top-5 variance explained:")

from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(embeddings)
for i, v in enumerate(pca.explained_variance_ratio_):
    print(f"  pc{i+1}: {v*100:.1f}%")

print("\nok — model works, forward pass produces 2048-dim embeddings")
