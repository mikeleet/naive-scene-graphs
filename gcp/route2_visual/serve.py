import os, json, io, zipfile, torch, numpy as np
from PIL import Image as PILImage
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torchvision

app = FastAPI()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = os.getenv("MODEL_TYPE", "pretrained")
checkpoint_gcs = os.getenv("CHECKPOINT_GCS", "")

if model_type == "pretrained":
    model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
    model = nn.Sequential(*list(model.children())[:-1])  # remove head
else:
    model = torchvision.models.resnet50(weights=None)
    if checkpoint_gcs and os.path.exists("/app/best_checkpoint.pth"):
        model.load_state_dict(torch.load("/app/best_checkpoint.pth"))

model = model.to(DEVICE).eval()

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class EmbedRequest(BaseModel):
    image_url: str = ""
    boxes: list[dict]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


@app.post("/embed")
def embed(req: EmbedRequest):
    if not req.boxes:
        raise HTTPException(400, "no boxes provided")

    try:
        if req.image_url:
            import urllib.request
            data = urllib.request.urlopen(req.image_url, timeout=10).read()
            img = PILImage.open(io.BytesIO(data)).convert("RGB")
        else:
            img = PILImage.new("RGB", (1024, 1024))
    except Exception as e:
        raise HTTPException(400, f"failed to load image: {e}")

    results = []
    for box in req.boxes:
        x1 = max(0, int(box.get("x1", 0)))
        y1 = max(0, int(box.get("y1", 0)))
        x2 = int(box.get("x2", img.width))
        y2 = int(box.get("y2", img.height))
        crop = img.crop((x1, y1, x2, y2))
        tensor = transform(crop).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            emb = model(tensor).squeeze().cpu().numpy()
        results.append([float(v) for v in emb])

    return EmbedResponse(embeddings=results)


@app.get("/health")
def health():
    return {"model_type": model_type, "device": str(DEVICE)}
