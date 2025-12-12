# per_class_report_ml.py
# Class-wise metrics (precision, recall, f1, support) for ML baselines on bottleneck features.

import json
import numpy as np
from pathlib import Path
from glob import glob
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# ---------- Config ----------
DATA_DIR = Path("data_tomato_splits")
IMG_SIZE = (224, 224)
SEED = 42
CACHE_DIR = Path("features_cache")  # speeds up re-runs
CACHE_DIR.mkdir(exist_ok=True)

# ---------- Helpers ----------
def load_split_paths(root: Path):
    """Return (paths, labels, class_names) for a split folder (sorted class order)."""
    class_names = sorted([d.name for d in root.iterdir() if d.is_dir()])
    paths, labels = [], []
    for i, cname in enumerate(class_names):
        for p in glob(str(root / cname / "*")):
            if Path(p).suffix.lower() in [".jpg", ".jpeg", ".png"]:
                paths.append(p); labels.append(i)
    return paths, np.array(labels), class_names

def read_img(p: str):
    return np.array(Image.open(p).convert("RGB").resize(IMG_SIZE), dtype=np.float32)

def extract_features(paths, base_model, batch_size=64, cache_key=None):
    """Extract MobileNetV2 bottleneck features, optionally cached."""
    cache_file = CACHE_DIR / f"{cache_key}.npy" if cache_key else None
    if cache_file and cache_file.exists():
        return np.load(cache_file)
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc=f"Extracting {cache_key or 'feats'}"):
        batch = np.stack([read_img(p) for p in paths[i:i+batch_size]], 0)
        batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
        f = base_model.predict(batch, verbose=0)
        f = tf.keras.layers.GlobalAveragePooling2D()(tf.convert_to_tensor(f)).numpy()
        feats.append(f)
    feats = np.concatenate(feats, 0)
    if cache_file:
        np.save(cache_file, feats)
    return feats

def save_report_csv(report_dict, class_names, out_csv):
    """Keeps only per-class rows + accuracy/macro/weighted averages, writes CSV."""
    df = pd.DataFrame(report_dict).T
    ordered = class_names + ["accuracy", "macro avg", "weighted avg"]
    df = df.reindex(ordered)
    df.to_csv(out_csv, float_format="%.4f")
    return df

# ---------- Load paths/labels ----------
tr_p, tr_y, class_names_tr = load_split_paths(DATA_DIR / "train")
va_p, va_y, _             = load_split_paths(DATA_DIR / "val")
te_p, te_y, class_names   = load_split_paths(DATA_DIR / "test")
assert class_names_tr == class_names, "Class order mismatch. Re-run splits or unify class ordering."

# ---------- Feature extractor ----------
base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=IMG_SIZE+(3,))
base.trainable = False

# ---------- Extract (with caching) ----------
tr_X = extract_features(tr_p, base, cache_key="train_X")
va_X = extract_features(va_p, base, cache_key="val_X")
te_X = extract_features(te_p, base, cache_key="test_X")

# Save class names once (for reference)
(Path(CACHE_DIR) / "class_names.json").write_text(json.dumps(class_names, indent=2))

# ---------- Scaling (for SVM/KNN) ----------
sc = StandardScaler()
tr_Xs = sc.fit_transform(tr_X)
va_Xs = sc.transform(va_X)
te_Xs = sc.transform(te_X)

# ---------- Train+Eval per model ----------
results_summary = []  # overall accuracy per model
reports_out = {
    "svm_rbf": ("SVM (RBF)",  SVC(kernel="rbf", C=10, gamma="scale", probability=True), True),
    "random_forest": ("RandomForest", RandomForestClassifier(n_estimators=300, random_state=SEED), False),
    "knn_k5": ("KNN (k=5)", KNeighborsClassifier(n_neighbors=5), True),
    "decision_tree": ("DecisionTree", DecisionTreeClassifier(random_state=SEED), False),
}

for key, (name, clf, use_scaled) in reports_out.items():
    if use_scaled:
        Xtr, Xva, Xte = tr_Xs, va_Xs, te_Xs
    else:
        Xtr, Xva, Xte = tr_X, va_X, te_X

    # Fit on train + val
    clf.fit(np.vstack([Xtr, Xva]), np.concatenate([tr_y, va_y]))

    # Predict on test
    y_pred = clf.predict(Xte)

    # Accuracy
    acc = accuracy_score(te_y, y_pred)
    results_summary.append((name, acc))

    # Classification report (per class)
    report = classification_report(te_y, y_pred, target_names=class_names, output_dict=True, digits=4)
    out_csv = f"per_class_{key}.csv"
    df = save_report_csv(report, class_names, out_csv)

    # Console output
    print(f"\n==== {name} ====")
    print(f"Accuracy: {acc*100:.2f}%  |  Misclassification: {100-acc*100:.2f}%")
    print(df[["precision","recall","f1-score","support"]])
    print(f"💾 Saved: {out_csv}")

# ---------- Summary CSV ----------
pd.DataFrame(
    [{"Model": m, "Accuracy (%)": f"{a*100:.2f}", "Misclassification (%)": f"{(1-a)*100:.2f}"} for m,a in results_summary]
).to_csv("ml_overall_summary.csv", index=False)
print("\n✅ Saved overall summary: ml_overall_summary.csv")
