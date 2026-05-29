from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import numpy as np
import pickle
import os
import sys

app = FastAPI(title="naive scene graph predictor")

model_dir = os.path.join(os.path.dirname(__file__), "model")

try:
    exist_clf = pickle.load(open(f"{model_dir}/exist_clf.pkl", "rb"))
    pred_clf = pickle.load(open(f"{model_dir}/pred_clf.pkl", "rb"))
    config = pickle.load(open(f"{model_dir}/config.pkl", "rb"))
except FileNotFoundError as e:
    print(f"model files not found in {model_dir}. run export_model.py first.")
    sys.exit(1)

OBJ2IDX: dict[str, int] = config["obj2idx"]
PRED2IDX: dict[str, int] = config["pred2idx"]
PRED_NAMES: list[str] = config["pred_names"]
OBJ_NAMES: list[str] = config["obj_names"]
MAX_PAIRS = 1000
MAX_BATCH_SIZE = 100


class Pair(BaseModel):
    head_class: str
    tail_class: str
    head_x1: int
    head_y1: int
    head_x2: int
    head_y2: int
    tail_x1: int
    tail_y1: int
    tail_x2: int
    tail_y2: int

    @field_validator("head_x1", "head_y1", "tail_x1", "tail_y1")
    @classmethod
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("box coordinates must be >= 0")
        return v

    @field_validator("head_x2", "head_y2", "tail_x2", "tail_y2")
    @classmethod
    def positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("box width/height must be > 0")
        return v


class PredictRequest(BaseModel):
    image_width: int = 1024
    image_height: int = 1024
    pairs: list[Pair]

    @field_validator("pairs")
    @classmethod
    def not_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("at least one pair required")
        if len(v) > MAX_PAIRS:
            raise ValueError(f"max {MAX_PAIRS} pairs per request")
        return v

    @field_validator("image_width", "image_height")
    @classmethod
    def positive_dims(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("image dimensions must be > 0")
        if v > 10000:
            raise ValueError("image dimensions must be <= 10000")
        return v


class Prediction(BaseModel):
    rank: int
    head: str
    predicate: str
    tail: str
    confidence: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]


class BatchRequest(BaseModel):
    items: list[PredictRequest]

    @field_validator("items")
    @classmethod
    def not_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("at least one batch item required")
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(f"max {MAX_BATCH_SIZE} items per batch")
        total_pairs = sum(len(item.pairs) for item in v)
        if total_pairs > MAX_PAIRS:
            raise ValueError(f"total pairs across batch exceeds {MAX_PAIRS}")
        return v


class BatchResponse(BaseModel):
    results: list[PredictResponse]


def extract_features(pair: Pair, img_w: int, img_h: int) -> list[int]:
    from shapely import equals, intersects, touches, contains
    from shapely.geometry import box as b
    import math

    if pair.head_class not in OBJ2IDX:
        raise HTTPException(status_code=400,
            detail=f"unknown head class '{pair.head_class}'")
    if pair.tail_class not in OBJ2IDX:
        raise HTTPException(status_code=400,
            detail=f"unknown tail class '{pair.tail_class}'")
    if pair.head_x2 <= pair.head_x1 or pair.head_y2 <= pair.head_y1:
        raise HTTPException(status_code=400,
            detail="head box invalid: x2 > x1 and y2 > y1 required")
    if pair.tail_x2 <= pair.tail_x1 or pair.tail_y2 <= pair.tail_y1:
        raise HTTPException(status_code=400,
            detail="tail box invalid: x2 > x1 and y2 > y1 required")
    if pair.head_x2 > img_w or pair.head_y2 > img_h:
        raise HTTPException(status_code=400,
            detail=f"head box exceeds image bounds ({img_w}x{img_h})")
    if pair.tail_x2 > img_w or pair.tail_y2 > img_h:
        raise HTTPException(status_code=400,
            detail=f"tail box exceeds image bounds ({img_w}x{img_h})")

    h_cls = OBJ2IDX[pair.head_class]
    t_cls = OBJ2IDX[pair.tail_class]

    hb = b(pair.head_x1, pair.head_y1, pair.head_x2, pair.head_y2)
    tb = b(pair.tail_x1, pair.tail_y1, pair.tail_x2, pair.tail_y2)

    if equals(hb, tb): topo = 7
    elif not intersects(hb, tb): topo = 1 if touches(hb, tb) else 0
    elif contains(hb, tb): topo = 3
    elif contains(tb, hb): topo = 4
    else: topo = 2

    hx = (pair.head_x1 + pair.head_x2) / 2
    hy = (pair.head_y1 + pair.head_y2) / 2
    tx = (pair.tail_x1 + pair.tail_x2) / 2
    ty = (pair.tail_y1 + pair.tail_y2) / 2
    dx, dy = tx - hx, ty - hy
    if dx == 0 and dy == 0: angle = 0
    else:
        rad = math.atan2(dy, dx)
        if rad < 0: rad += 2 * math.pi
        angle = int((rad + math.pi / 8) // (math.pi / 4)) % 8

    h_area = (pair.head_x2 - pair.head_x1) * (pair.head_y2 - pair.head_y1)
    t_area = (pair.tail_x2 - pair.tail_x1) * (pair.tail_y2 - pair.tail_y1)

    def ratio_bin(ratio, lo, hi):
        if ratio <= 0: return int(lo)
        v = round(math.log(ratio))
        return max(int(lo), min(int(hi), v))

    return [h_cls, t_cls, topo, angle,
            ratio_bin(h_area / max(t_area, 1), 0, 18),
            ratio_bin(h_area / max(img_w * img_h, 1), 0, 9),
            ratio_bin(t_area / max(img_w * img_h, 1), 0, 9)]


def _run_inference(features):
    X = np.array(features, dtype=np.int32)
    exist_logp = exist_clf.predict_log_proba(X)[:, 1]
    pred_logp = pred_clf.predict_log_proba(X)
    best_pred = np.argmax(pred_logp, axis=1)
    best_conf = pred_logp[np.arange(len(X)), best_pred]
    combined = exist_logp + best_conf
    ranking = np.argsort(combined)[::-1]
    return ranking, best_pred, combined, features


def _features_to_response(features, ranking, best_pred, combined):
    results = []
    for rank, idx in enumerate(ranking):
        results.append(Prediction(
            rank=rank,
            head=OBJ_NAMES[features[idx][0]],
            predicate=PRED_NAMES[best_pred[idx]],
            tail=OBJ_NAMES[features[idx][1]],
            confidence=float(combined[idx]),
        ))
    return PredictResponse(predictions=results)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        features = [extract_features(p, req.image_width, req.image_height)
                    for p in req.pairs]
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"feature extraction failed: {e}")
    try:
        ranking, best_pred, combined, features = _run_inference(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"model inference failed: {e}")
    return _features_to_response(features, ranking, best_pred, combined)


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    all_features, pair_counts = [], []
    for item in req.items:
        try:
            feats = [extract_features(p, item.image_width, item.image_height)
                     for p in item.pairs]
        except HTTPException: raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"feature extraction failed: {e}")
        all_features.extend(feats)
        pair_counts.append(len(feats))

    try:
        ranking, best_pred, combined, _ = _run_inference(all_features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"model inference failed: {e}")

    results, offset = [], 0
    for count in pair_counts:
        item_features = all_features[offset:offset + count]
        item_ranking = [idx for idx in ranking if offset <= idx < offset + count]
        item_results = []
        for rank, idx in enumerate(item_ranking):
            item_results.append(Prediction(
                rank=rank,
                head=OBJ_NAMES[item_features[idx - offset][0]],
                predicate=PRED_NAMES[best_pred[idx]],
                tail=OBJ_NAMES[item_features[idx - offset][1]],
                confidence=float(combined[idx]),
            ))
        results.append(PredictResponse(predictions=item_results))
        offset += count
    return BatchResponse(results=results)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "sklearn categoricalnb",
        "classes": 150,
        "training_scale": "boxes at 1024x1024 — pass consistent coords for best results",
    }


@app.get("/model_info")
def model_info():
    return {
        "object_classes": len(OBJ_NAMES),
        "predicate_classes": len(PRED_NAMES),
        "sample_objects": OBJ_NAMES[:10],
        "sample_predicates": PRED_NAMES[:10],
        "feature_counts": config["feature_n_categories"],
        "exist_classifier_features": len(exist_clf.feature_log_prob_),
        "pred_classifier_features": len(pred_clf.feature_log_prob_),
    }
