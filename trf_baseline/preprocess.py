import json
import pickle
from pathlib import Path

import h5py
import numpy as np

from trf_baseline.features import extract


def run(data_dir: Path | None = None):
    if data_dir is None:
        for candidate in [Path("/app/data"), Path(__file__).resolve().parent.parent / "data"]:
            if candidate.exists():
                data_dir = candidate
                break
        else:
            raise FileNotFoundError("data dir not found")

    out = data_dir / "preprocessed"
    out.mkdir(parents=True, exist_ok=True)

    if (out / "train_X.npy").exists():
        print("preprocessed data already exists, skipping")
        return

    with open(data_dir / "vg150" / "VG-SGG-dicts.json") as f:
        d = json.load(f)
    idx_to_label = {int(k): v for k, v in d["idx_to_label"].items()}
    idx_to_pred = {int(k): v for k, v in d["idx_to_predicate"].items()}

    with h5py.File(data_dir / "vg150" / "VG-SGG.h5", "r") as f:
        boxes = f["boxes_1024"][:]
        labels = f["labels"][:].flatten()
        rels = f["relationships"][:]
        preds = f["predicates"][:].flatten()
        img_to_first = f["img_to_first_box"][:]
        img_to_last = f["img_to_last_box"][:]
        img_to_first_rel = f["img_to_first_rel"][:]
        img_to_last_rel = f["img_to_last_rel"][:]
        splits = f["split"][:]

    xc, yc, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    boxes_corner = np.column_stack([
        (xc - w / 2).astype(np.int32), (yc - h / 2).astype(np.int32),
        (xc + w / 2).astype(np.int32), (yc + h / 2).astype(np.int32),
    ])

    for split_val, split_name in [(0, "train"), (2, "test")]:
        print(f"processing {split_name} split...")

        all_X, all_y, all_e = [], [], []
        all_pairs = []  # (img_idx, head_local, tail_local, h_cls, t_cls)
        total_pairs = 0
        total_rels = 0

        for img_idx in range(len(img_to_first)):
            if splits[img_idx] != split_val:
                continue

            box_start = img_to_first[img_idx]
            box_end = img_to_last[img_idx] + 1
            if box_end <= box_start:
                continue

            img_boxes = boxes_corner[box_start:box_end]
            img_labels = labels[box_start:box_end]
            n_obj = len(img_boxes)
            if n_obj < 2:
                continue

            rel_start = img_to_first_rel[img_idx]
            rel_end = img_to_last_rel[img_idx] + 1
            if rel_end <= rel_start:
                continue

            img_rels = rels[rel_start:rel_end]
            img_preds = preds[rel_start:rel_end]

            # triplet ground truth: (head_local, tail_local, pred_idx)
            gt_triplets = set()
            for (sid, oid), pid in zip(img_rels, img_preds):
                subj_local = sid - box_start
                obj_local = oid - box_start
                if 0 <= subj_local < n_obj and 0 <= obj_local < n_obj:
                    gt_triplets.add((int(subj_local), int(obj_local), int(pid)))

            img_pairs = []
            for i in range(n_obj):
                hi = {"x1": int(img_boxes[i][0]), "y1": int(img_boxes[i][1]),
                      "x2": int(img_boxes[i][2]), "y2": int(img_boxes[i][3])}
                h_cls = int(img_labels[i])
                if h_cls >= len(idx_to_label):
                    continue
                for j in range(n_obj):
                    if i == j:
                        continue
                    tj = {"x1": int(img_boxes[j][0]), "y1": int(img_boxes[j][1]),
                          "x2": int(img_boxes[j][2]), "y2": int(img_boxes[j][3])}
                    t_cls = int(img_labels[j])
                    if t_cls >= len(idx_to_label):
                        continue

                    feats = extract(hi, tj, 1024, 1024, h_cls, t_cls)
                    pid = -1
                    for gt in gt_triplets:
                        if gt[0] == i and gt[1] == j:
                            pid = gt[2]
                            break

                    all_X.append(feats)
                    all_y.append(pid)
                    all_e.append(pid >= 0)
                    img_pairs.append((i, j, h_cls, t_cls, pid))
                    total_pairs += 1
                    if pid >= 0:
                        total_rels += 1

            all_pairs.append(img_pairs)

        X = np.array(all_X, dtype=np.int16)
        y = np.array(all_y, dtype=np.int16)
        e = np.array(all_e, dtype=bool)

        np.save(out / f"{split_name}_X.npy", X)
        np.save(out / f"{split_name}_y.npy", y)
        np.save(out / f"{split_name}_exist.npy", e)

        with open(out / f"{split_name}_pairs.pkl", "wb") as f:
            pickle.dump(all_pairs, f)

        print(f"  {split_name}: {len(all_pairs)} images, {total_pairs} pairs, "
              f"{total_rels} relationships ({100*total_rels/max(total_pairs,1):.1f}%)")

    print(f"\ndone. arrays saved to {out}")


if __name__ == "__main__":
    run()
