# naive-scene-graphs

7-feature naive bayes, no pixels, just box metadata. competitive with deep
scene graph generators.

built same baseline 3 ways. then gcp.

- nlp: https://ruder.io/nlp-imagenet/ — hypothesis-only baseline, 67% from shortcuts
- gbt: relational features > raw, xgboost basket > individual, greedy split = greedy descent?

## approach

cpu: sklearn catnb / python:3.11-slim / macos m4 ✓
deploy: fastapi + docker / python:3.11-slim ✓
gcp: cloud run + terraform / route1 train, route2 3-endpoint deploy

### platform notes

- **m4**: multi-arch pull, native arm64. preprocess + sklearn here
- **4090**: wsl2 + nvidia toolkit, nvidia-smi smoke test in dockerfile
- **cloud run**: same slim base, ~150mb image, <2s cold start

## findings

to be filled after m9.

## notebooks

install deps: `pip install -r requirements-notebooks.txt`
start: `jupyter notebook scratch/`
symlink: `scratch/data → ../data` — run notebooks from scratch/ directory, H5 loads via relative path

all notebooks use `DATA = "data"` — access h5 + dicts + preprocessed arrays via symlink

## data

vg images (for gpu training): `../img/images.zip` + `../img/images2.zip`
don't extract — read from zip. script: `scratch/image_preview.ipynb`

h5 + dicts + mapping: `scratch/data` symlink → `../data`

## progress

### m1 — scaffold ✓
dirs, docker compose, makefile

### m2 — dataprep ✓
migrated vg150 split from https://github.com/danfeiX/scene-graph-TF-release
dataloader from https://github.com/rowanz/neural-motifs
7 features as listed in the paper: class, topology, angle, area ratios
(no public code for these — just the paper's description). categoricalnb.

### m3 — cpu ✓
sklearn, 2 classifiers (exist + pred). 1.8s train, 4.7s infer, 10M train pairs,
5M test pairs. R@100=57.62, mR@100=20.50 (predcl). close to paper.

### m4 — paper match ✓
all 3 feature combos within 3% of paper. details: [docs/paper-match.ipynb](docs/paper-match.ipynb)

### m5 — docker prediction service ✓
fastapi endpoint, exported sklearn model, input validation + batch mode.
test.sh covers health, predict, 5 error cases, batch. pytorch skipped —
model trains in 1.8s. gpu conversion makes more sense on gcp where infra
already handles gpu inference.

### m5.5 — crop preprocessing (running)
1.1M box crops from v1.2 annotations via vertex ai cpu job.
zips on local ssd, python gcs client for upload, 4 workers parallel,
psutil resource monitoring. crops saved as tar archives on gcs,
train/test split via h5 split masks. output: crops_train_w*.tar +
crops_test_w*.tar + index_train.jsonl + index_test.jsonl + class_weights.json

### m6 — gcp route 1 (training)
resnet-50 from scratch on vg150 crops. vertex ai t4 gpu. full logs,
tensorboard, hyperparameter tracking. outputs best_checkpoint.pth.

### m7 — gcp route 2 (3-model deploy)
3 cloud run endpoints: nb, visual-pretrained, visual-trained.
terraform: 5 modules (registry, nb, visual×2, vertex ai, iam).
combined prediction script compares geometric vs visual accuracy.

### m8 — benchmark + results
populate table, final numbers

## checklist

- [x] download vg150 data (h5 + dicts + v1.2 relationships)
- [x] 7-feature extraction (shapely topology)
- [x] scratch notebooks — data exploration, topology tests, mem, image preview
- [x] docker scaffold + makefile
- [x] sklearn categoricalnb baseline (1.8s train, 10M pairs)
- [x] paper match — all 3 feature combos within 3%
- [x] image preview — real vg photos + v1.2 boxes + relationship arrows
- [x] docker prediction service — fastapi + exported model, test.sh passes
- [ ] crop preprocessing — vertex ai cpu, 1.1M crops → gcs tar archives (running)
- [ ] gcp route 1 — resnet-50 training on vertex ai
- [ ] gcp route 2 — 3-model cloud run deploy + terraform
- [ ] benchmark table populated
- [ ] questions for interview
