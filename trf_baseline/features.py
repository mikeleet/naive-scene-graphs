import math
import numpy as np
from shapely import equals, intersects, touches, contains
from shapely.geometry import box as _box


def _bbox_area(box: dict) -> float:
    return (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])


def _topology(head: dict, tail: dict) -> int:
    hb = _box(head["x1"], head["y1"], head["x2"], head["y2"])
    tb = _box(tail["x1"], tail["y1"], tail["x2"], tail["y2"])

    if equals(hb, tb):
        return 7
    if not intersects(hb, tb):
        return 1 if touches(hb, tb) else 0
    if contains(hb, tb):
        return 3
    if contains(tb, hb):
        return 4
    return 2


def _angle(head: dict, tail: dict) -> int:
    hx = (head["x1"] + head["x2"]) / 2
    hy = (head["y1"] + head["y2"]) / 2
    tx = (tail["x1"] + tail["x2"]) / 2
    ty = (tail["y1"] + tail["y2"]) / 2

    dx = tx - hx
    dy = ty - hy
    if dx == 0 and dy == 0:
        return 0

    rad = math.atan2(dy, dx)
    if rad < 0:
        rad += 2 * math.pi
    return int((rad + math.pi / 8) // (math.pi / 4)) % 8


def _ratio_bin_rel(ratio: float) -> int:
    if ratio <= 0:
        return 0
    v = round(math.log(ratio))
    clipped = max(-9, min(9, v))
    return clipped + 9


def _ratio_bin_area(ratio: float) -> int:
    if ratio <= 0:
        return 0
    v = round(math.log(ratio))
    return max(0, min(9, v))


def extract(
    head: dict,
    tail: dict,
    img_w: int,
    img_h: int,
    head_cls: int,
    tail_cls: int,
) -> list[int]:
    h_area = _bbox_area(head)
    t_area = _bbox_area(tail)
    return [
        head_cls,
        tail_cls,
        _topology(head, tail),
        _angle(head, tail),
        _ratio_bin_rel(h_area / max(t_area, 1)),
        _ratio_bin_area(h_area / max(img_w * img_h, 1)),
        _ratio_bin_area(t_area / max(img_w * img_h, 1)),
    ]


def from_image(
    image: dict,
    class_lookup: dict[str, int],
    pred_lookup: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    objs = image["objects"]
    n = len(objs)

    obj_names = {}
    for o in objs:
        names = o.get("names", [o.get("name", "")])
        obj_names[o["object_id"]] = names[0] if names else o.get("name", "")

    rel_map = {}
    for r in image["relationships"]:
        key = (r["subject_id"], r["object_id"])
        p = pred_lookup.get(r["predicate"])
        if p is not None:
            rel_map[key] = p

    pairs = []
    for i, hi in enumerate(objs):
        h_cls = class_lookup.get(obj_names[hi["object_id"]])
        if h_cls is None:
            continue
        for j, ti in enumerate(objs):
            if i == j:
                continue
            t_cls = class_lookup.get(obj_names[ti["object_id"]])
            if t_cls is None:
                continue
            feat = extract(hi, ti, image["width"], image["height"], h_cls, t_cls)
            pair = (feat, rel_map.get((hi["object_id"], ti["object_id"]), -1))
            pairs.append(pair)

    X = np.array([p[0] for p in pairs], dtype=np.int16)
    y = np.array([p[1] for p in pairs], dtype=np.int16)
    exist = y >= 0
    return X, y, exist
