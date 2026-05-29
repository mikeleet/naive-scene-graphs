import pickle, numpy as np, os, sys

from sklearn.naive_bayes import CategoricalNB

data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed")
model_dir = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(model_dir, exist_ok=True)

X_train = np.load(os.path.join(data_dir, "train_X.npy")).astype(np.int32)
y_train = np.load(os.path.join(data_dir, "train_y.npy")).astype(np.int32)
exist_train = np.load(os.path.join(data_dir, "train_exist.npy")).astype(bool)

FEATURE_N_CATEGORIES = [150, 150, 8, 8, 19, 10, 10]

print("training...")
exist_clf = CategoricalNB(alpha=1.0, min_categories=2)
for i in range(0, len(X_train), 500000):
    exist_clf.partial_fit(X_train[i:i+500000], exist_train[i:i+500000], classes=[0, 1])

pos = exist_train
pred_clf = CategoricalNB(alpha=1.0, min_categories=FEATURE_N_CATEGORIES)
for i in range(0, pos.sum(), 500000):
    pred_clf.partial_fit(X_train[pos][i:i+500000], y_train[pos][i:i+500000], classes=list(range(50)))

pickle.dump(exist_clf, open(os.path.join(model_dir, "exist_clf.pkl"), "wb"))
pickle.dump(pred_clf, open(os.path.join(model_dir, "pred_clf.pkl"), "wb"))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trf_baseline.config import OBJ2IDX, PRED2IDX, OBJECT_CLASSES, PREDICATES

config = {
    "feature_n_categories": FEATURE_N_CATEGORIES,
    "obj2idx": OBJ2IDX,
    "pred2idx": PRED2IDX,
    "pred_names": PREDICATES,
    "obj_names": OBJECT_CLASSES,
}
pickle.dump(config, open(os.path.join(model_dir, "config.pkl"), "wb"))
print(f"saved to {model_dir}/")
