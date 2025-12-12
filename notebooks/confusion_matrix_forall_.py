# confusion_matrices_all.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from glob import glob
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# ---------- Config ----------
DATA_DIR = Path("data_tomato_splits")
MODEL_PATH = "mobilenetv2_tomato.h5"
IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42

# ---------- Helpers ----------
def load_split_paths(root: Path):
    """Return (paths, labels, class_names) for a split folder."""
    class_names = sorted([d.name for d in root.iterdir() if d.is_dir()])
    paths, labels = [], []
    for i, cname in enumerate(class_names):
        for p in glob(str(root / cname / "*")):
            if Path(p).suffix.lower() in [".jpg", ".jpeg", ".png"]:
                paths.append(p); labels.append(i)
    return paths, np.array(labels), class_names

def read_img(p: str):
    # PIL is robust to different encodings
    return np.array(Image.open(p).convert("RGB").resize(IMG_SIZE), dtype=np.float32)

def extract_features(paths, base_model, batch_size=64):
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Extracting bottleneck"):
        batch = np.stack([read_img(p) for p in paths[i:i+batch_size]], 0)
        batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
        f = base_model.predict(batch, verbose=0)
        f = tf.keras.layers.GlobalAveragePooling2D()(tf.convert_to_tensor(f)).numpy()
        feats.append(f)
    return np.concatenate(feats, 0)

def pretty_confusion_matrix(y_true, y_pred, class_names, title, out_png, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    # Build annotation with counts + %
    ann = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ann[i, j] = f"{cm[i,j]}\n({cm_norm[i,j]*100:0.1f}%)"
    plt.figure(figsize=(10.5, 8))
    sns.set_style("whitegrid"); sns.set(font_scale=1.0)
    ax = sns.heatmap(cm_norm if normalize else cm,
                     annot=ann, fmt="", cmap="YlGnBu",
                     xticklabels=class_names, yticklabels=class_names,
                     cbar_kws={'label': 'Proportion' if normalize else 'Count'})
    plt.title(title, fontsize=14, weight='bold', pad=16)
    plt.xlabel("Predicted", fontsize=12); plt.ylabel("True", fontsize=12)
    plt.xticks(rotation=35, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    # Console report
    print(f"\n{title}")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print(f"✅ Saved: {out_png}")

# ---------- CNN (MobileNetV2) ----------
print("==> Evaluating CNN (MobileNetV2)")
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="categorical",
    shuffle=False
)
cnn_class_names = test_ds.class_names
cnn_model = tf.keras.models.load_model(MODEL_PATH)

# y_true
y_true_cnn = np.concatenate([y.numpy() for x, y in test_ds])
y_true_cnn = np.argmax(y_true_cnn, axis=1)

# y_pred
y_pred_cnn = []
for x, _ in test_ds:
    preds = cnn_model.predict(x, verbose=0)
    y_pred_cnn.append(np.argmax(preds, axis=1))
y_pred_cnn = np.concatenate(y_pred_cnn)

pretty_confusion_matrix(
    y_true_cnn, y_pred_cnn, cnn_class_names,
    title="Confusion Matrix — CNN (MobileNetV2)",
    out_png="cm_cnn_mobilenet.png",
    normalize=True
)

# ---------- ML models on bottleneck features ----------
print("\n==> Evaluating ML models on MobileNetV2 bottleneck features")
# Load paths/labels
tr_p, tr_y, class_names_tr = load_split_paths(DATA_DIR / "train")
va_p, va_y, _             = load_split_paths(DATA_DIR / "val")
te_p, te_y, class_names   = load_split_paths(DATA_DIR / "test")
assert class_names == cnn_class_names, "Class order mismatch. Re-run splits consistently."

# Feature extractor base
base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,))
base.trainable = False

# Extract features
tr_X = extract_features(tr_p, base)
va_X = extract_features(va_p, base)
te_X = extract_features(te_p, base)

# Scale for distance-based models
sc = StandardScaler()
tr_Xs = sc.fit_transform(tr_X)
va_Xs = sc.transform(va_X)
te_Xs = sc.transform(te_X)

def fit_predict_and_plot(name, clf, use_scaled=True, file_stub=""):
    if use_scaled:
        Xtr, Xva, Xte = tr_Xs, va_Xs, te_Xs
    else:
        Xtr, Xva, Xte = tr_X, va_X, te_X
    clf.fit(np.vstack([Xtr, Xva]), np.concatenate([tr_y, va_y]))
    pred = clf.predict(Xte)
    pretty_confusion_matrix(
        te_y, pred, class_names,
        title=f"Confusion Matrix — {name}",
        out_png=f"cm_{file_stub}.png",
        normalize=True
    )

# SVM (scaled)
fit_predict_and_plot("SVM (RBF)", SVC(kernel="rbf", C=10, gamma="scale"), use_scaled=True,  file_stub="svm_rbf")
# Random Forest (unscaled features fine)
fit_predict_and_plot("Random Forest", RandomForestClassifier(n_estimators=300, random_state=SEED), use_scaled=False, file_stub="random_forest")
# KNN (scaled)
fit_predict_and_plot("KNN (k=5)", KNeighborsClassifier(n_neighbors=5), use_scaled=True, file_stub="knn_k5")
# Decision Tree (unscaled)
fit_predict_and_plot("Decision Tree", DecisionTreeClassifier(random_state=SEED), use_scaled=False, file_stub="decision_tree")

print("\nAll confusion matrices saved:")
print(" - cm_cnn_mobilenet.png")
print(" - cm_svm_rbf.png")
print(" - cm_random_forest.png")
print(" - cm_knn_k5.png")
print(" - cm_decision_tree.png")
