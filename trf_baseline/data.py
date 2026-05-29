import json
import zipfile
from pathlib import Path


def load(data_dir: Path | None = None) -> list[dict]:
    if data_dir is None:
        for candidate in [Path("/app/data"), Path(__file__).resolve().parent.parent / "data"]:
            if candidate.exists():
                data_dir = candidate
                break
        else:
            raise FileNotFoundError("could not find data directory")

    ann_dir = data_dir / "vg150" / "annotations"
    json_path = ann_dir / "relationships.json"
    zip_path = ann_dir / "relationships.json.zip"

    if json_path.exists():
        with open(json_path) as f:
            raw = json.load(f)
    elif zip_path.exists():
        print(f"reading from zip: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.0f} mb)")
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open("relationships.json") as f:
                raw = json.load(f)
    else:
        raise FileNotFoundError(
            f"vg150 annotations not found at {json_path} or {zip_path}"
        )

    # zip format: list of {image_id, relationships}
    if isinstance(raw, list):
        return _parse_list(raw)

    # legacy vg dict format
    return _parse_dict(raw)


def _parse_list(raw: list) -> list[dict]:
    images = []
    for entry in raw:
        img = {
            "image_id": entry["image_id"],
            "width": 0,
            "height": 0,
            "objects": [],
            "relationships": [],
        }
        seen_objects: dict[int, dict] = {}

        for rel in entry.get("relationships", []):
            for side in ("subject", "object"):
                obj = rel.get(side, {})
                oid = obj.get("object_id")
                if oid is None:
                    continue
                if oid not in seen_objects:
                    w = obj.get("w", 0)
                    h = obj.get("h", 0)
                    x = obj.get("x", 0)
                    y = obj.get("y", 0)
                    name = obj.get("name", "unknown")
                    names = obj.get("names", [name] if name else ["unknown"])
                    seen_objects[oid] = {
                        "object_id": oid,
                        "name": name if isinstance(name, str) and name
                                 else (names[0] if names else "unknown"),
                        "x1": x, "y1": y,
                        "x2": x + w, "y2": y + h,
                        "w": w, "h": h,
                    }
                    img["width"] = max(img["width"], x + w)
                    img["height"] = max(img["height"], y + h)

            sid = rel.get("subject", {}).get("object_id")
            oid = rel.get("object", {}).get("object_id")
            if sid is not None and oid is not None:
                img["relationships"].append({
                    "subject_id": sid,
                    "predicate": rel["predicate"].lower(),
                    "object_id": oid,
                })

        img["objects"] = list(seen_objects.values())
        if img["objects"] and img["relationships"]:
            images.append(img)

    print(f"loaded {len(images)} images with relationships")
    return images


def _parse_dict(raw: dict) -> list[dict]:
    objects_by_id: dict[int, list[dict]] = {}
    for obj in raw.get("objects", []):
        obj_id = obj["object_id"]
        objects_by_id.setdefault(obj_id, []).append(obj)

    images: dict[int, dict] = {}
    for img_meta in raw.get("image_data", []):
        img_id = img_meta["image_id"]
        images[img_id] = {
            "image_id": img_id,
            "width": img_meta.get("width", 0),
            "height": img_meta.get("height", 0),
            "objects": [],
            "relationships": [],
        }

    for synset in raw.get("object_synsets", []):
        img_id = synset["image_id"]
        if img_id not in images:
            continue
        for obj_id in synset.get("object_ids", []):
            if obj_id not in objects_by_id:
                continue
            for obj in objects_by_id[obj_id]:
                names = obj.get("names", [obj.get("name", "unknown")])
                name = names[0] if names else "unknown"
                w = obj.get("w", 0)
                h = obj.get("h", 0)
                x = obj.get("x", 0)
                y = obj.get("y", 0)
                images[img_id]["objects"].append({
                    "object_id": obj_id,
                    "name": name,
                    "x1": x, "y1": y,
                    "x2": x + w, "y2": y + h,
                    "w": w, "h": h,
                })

    for rel in raw.get("relationships", []):
        img_id = rel["image_id"]
        if img_id in images:
            images[img_id]["relationships"].append({
                "subject_id": rel["subject_id"],
                "predicate": rel["predicate"].lower(),
                "object_id": rel["object_id"],
            })

    result = [img for img in images.values() if img["relationships"] and img["objects"]]
    print(f"loaded {len(result)} images with relationships")
    return result
