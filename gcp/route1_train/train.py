import os, json, time, tarfile, io
import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
from PIL import Image as PILImage

HPARAMS = {
    "model": "resnet50",
    "init": "random",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 10,
    "weight_decay": 1e-4,
    "num_classes": 150,
    "num_workers": 4,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUCKET = os.getenv("BUCKET", "metal-ranger-123605-scene-graph")
RESULTS = "/app/results"
os.makedirs(RESULTS, exist_ok=True)
LOG_DIR = "/app/logs"
os.makedirs(LOG_DIR, exist_ok=True)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
])


class TarCropDataset(IterableDataset):
    def __init__(self, bucket_name, prefix, index_path, shuffle=True):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.shuffle = shuffle

        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        index_data = bucket.blob(index_path).download_as_string()
        self.entries = [json.loads(line) for line in
                        index_data.decode().strip().split("\n")
                        if line.strip()]

        self.tar_cache = {}

    def _get_tar(self, tar_name):
        if tar_name in self.tar_cache:
            return self.tar_cache[tar_name]

        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(f"crops/{tar_name}")
        data = blob.download_as_bytes()
        self.tar_cache[tar_name] = tarfile.open(
            fileobj=io.BytesIO(data), mode="r")
        return self.tar_cache[tar_name]

    def __iter__(self):
        import random
        entries = list(self.entries)
        if self.shuffle:
            random.shuffle(entries)

        for entry in entries:
            try:
                tar = self._get_tar(entry["tar"])
                member = tar.getmember(entry["crop"])
                img_bytes = tar.extractfile(member).read()
                img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                tensor = transform(img)
                label = entry["label"]
                yield tensor, label
            except Exception:
                continue

    def __len__(self):
        return len(self.entries)


def train_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f"  epoch {epoch} batch {batch_idx}: "
                  f"loss={loss.item():.3f} acc={100.*correct/total:.1f}%")

    return total_loss / max(len(loader), 1), 100. * correct / max(total, 1)


def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / max(len(loader), 1), 100. * correct / max(total, 1)


def main():
    print(f"hp: {json.dumps(HPARAMS, indent=2)}")
    print(f"device: {DEVICE}")

    print("loading datasets...")
    train_ds = TarCropDataset(BUCKET, "crops", "crops/index_train.jsonl",
                              shuffle=True)
    test_ds = TarCropDataset(BUCKET, "crops", "crops/index_test.jsonl",
                             shuffle=False)
    print(f"train: {len(train_ds):,} crops, test: {len(test_ds):,} crops")

    train_loader = DataLoader(train_ds, batch_size=HPARAMS["batch_size"],
                              num_workers=HPARAMS["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=HPARAMS["batch_size"],
                             num_workers=HPARAMS["num_workers"])

    print("building model...")
    model = torchvision.models.resnet50(weights=None,
                                         num_classes=HPARAMS["num_classes"])
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=HPARAMS["learning_rate"],
                                 weight_decay=HPARAMS["weight_decay"])

    metrics = []
    best_acc = 0
    t0 = time.time()

    for epoch in range(1, HPARAMS["epochs"] + 1):
        print(f"\n=== epoch {epoch}/{HPARAMS['epochs']} ===")
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, epoch)
        val_loss, val_acc = validate(model, test_loader, criterion)

        elapsed = (time.time() - t0) / 60
        print(f"  train: loss={train_loss:.3f} acc={train_acc:.2f}%")
        print(f"  test:  loss={val_loss:.3f} acc={val_acc:.2f}%")
        print(f"  time:  {elapsed:.1f} min")

        metrics.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "elapsed_min": round(elapsed, 1),
        })

        if val_acc > best_acc:
            best_acc = val_acc
            path = f"{RESULTS}/best_checkpoint.pth"
            torch.save(model.state_dict(), path)
            print(f"  saved best (acc={val_acc:.2f}%)")

    total_min = (time.time() - t0) / 60
    results = {
        "hyperparameters": HPARAMS,
        "best_val_acc": round(best_acc, 2),
        "total_time_min": round(total_min, 1),
        "device": str(DEVICE),
        "metrics": metrics,
        "train_crops": len(train_ds),
        "test_crops": len(test_ds),
    }

    with open(f"{RESULTS}/training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(f"{RESULTS}/metrics_per_epoch.csv", "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,elapsed_min\n")
        for m in metrics:
            f.write(f"{m['epoch']},{m['train_loss']},{m['train_acc']},"
                    f"{m['val_loss']},{m['val_acc']},{m['elapsed_min']}\n")

    print(f"\ndone. best acc: {best_acc:.2f}%, total: {total_min:.1f} min")


if __name__ == "__main__":
    main()
