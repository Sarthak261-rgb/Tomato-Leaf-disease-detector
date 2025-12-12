# roc_auc_all_models.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from glob import glob
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# ============ CONFIG ============
DATA_DIR = Path("data_tomato_splits")
MODEL_PATH = "mobilenetv2_tomato.h5"
IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42

sns.set_style("whitegrid")

# ============ HELPERS ============
def load_split_paths(root: Path):
    """Return (paths, labels, class_names) for a split folder."""
    class_names = sorted([d.name for d in root.iterdir() if d.is_dir()])
    paths, labels = [], []
    for i, cname in enumerate(class_names):
        for p in glob(str(root / cname / "*")):
            if Path(p).suffix.lower() in [".jpg", ".jpeg", ".png"]:
                paths.append(p)
                labels.append(i)
    return paths, np.array(labels), class_names


def read_img(p):
    return np.array(Image.open(p).convert("RGB").resize(IMG_SIZE), dtype=np.float32)


def extract_features(paths, base_model, batch_size=64):
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Extracting bottleneck"):
        batch = np.stack([read_img(p) for p in paths[i : i + batch_size]], 0)
        batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
        f = base_model.predict(batch, verbose=0)
        f = tf.keras.layers.GlobalAveragePooling2D()(tf.convert_to_tensor(f)).numpy()
        feats.append(f)
    return np.concatenate(feats, 0)


def plot_multiclass_roc(y_true, y_score, class_names, model_name, filename):
    """Plot micro/macro ROC curves for multi-class."""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr, tpr, roc_auc = dict(), dict(), dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    roc_auc["macro"] = auc(all_fpr, mean_tpr)

    # ---- Plot ----
    plt.figure(figsize=(8, 6))
    plt.plot(fpr["micro"], tpr["micro"],
             label=f"micro-average (AUC = {roc_auc['micro']:.3f})",
             color="deeppink", linestyle=":", linewidth=3)
    plt.plot(all_fpr, mean_tpr,
             label=f"macro-average (AUC = {roc_auc['macro']:.3f})",
             color="navy", linestyle=":", linewidth=3)

    for i, c in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], lw=1, alpha=0.6,
                 label=f"{c} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {filename}")


# ============ CNN (MobileNetV2) ============
print("==> Evaluating CNN (MobileNetV2)")
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="categorical",
    shuffle=False,
)
class_names = test_ds.class_names
cnn_model = tf.keras.models.load_model(MODEL_PATH)

y_true_cnn = np.concatenate([y.numpy() for x, y in test_ds])
y_true_cnn = np.argmax(y_true_cnn, axis=1)
y_score_cnn = []
for x, _ in test_ds:
    preds = cnn_model.predict(x, verbose=0)
    y_score_cnn.append(preds)
y_score_cnn = np.concatenate(y_score_cnn)

plot_multiclass_roc(y_true_cnn, y_score_cnn, class_names,
                    "CNN (MobileNetV2)", "roc_cnn_mobilenet.png")

# ============ ML MODELS (on bottleneck features) ============
print("\n==> Evaluating ML models for ROC-AUC")
tr_p, tr_y, _ = load_split_paths(DATA_DIR / "train")
va_p, va_y, _ = load_split_paths(DATA_DIR / "val")
te_p, te_y, _ = load_split_paths(DATA_DIR / "test")

base = tf.keras.applications.MobileNetV2(include_top=False,
                                         weights="imagenet",
                                         input_shape=IMG_SIZE + (3,))
base.trainable = False

tr_X = extract_features(tr_p, base)
va_X = extract_features(va_p, base)
te_X = extract_features(te_p, base)

sc = StandardScaler()
tr_Xs = sc.fit_transform(tr_X)
va_Xs = sc.transform(va_X)
te_Xs = sc.transform(te_X)

# --- ML models with predict_proba ---
def fit_predict_prob(name, clf, use_scaled=True, file_stub=""):
    Xtr, Xva, Xte = (tr_Xs, va_Xs, te_Xs) if use_scaled else (tr_X, va_X, te_X)
    clf.fit(np.vstack([Xtr, Xva]), np.concatenate([tr_y, va_y]))
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(Xte)
    else:  # SVM decision function
        y_score = clf.decision_function(Xte)
        # Convert to probability-like scale [0,1]
        y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())
    plot_multiclass_roc(te_y, y_score, class_names,
                        f"{name}", f"roc_{file_stub}.png")

fit_predict_prob("SVM (RBF)", SVC(kernel="rbf", C=10, gamma="scale",
                                  probability=True), use_scaled=True, file_stub="svm_rbf")
fit_predict_prob("Random Forest", RandomForestClassifier(n_estimators=300,
                                                         random_state=SEED),
                 use_scaled=False, file_stub="random_forest")
fit_predict_prob("KNN (k=5)", KNeighborsClassifier(n_neighbors=5),
                 use_scaled=True, file_stub="knn_k5")
fit_predict_prob("Decision Tree", DecisionTreeClassifier(random_state=SEED),
                 use_scaled=False, file_stub="decision_tree")

print("\nAll ROC curves saved:")
print(" - roc_cnn_mobilenet.png")
print(" - roc_svm_rbf.png")
print(" - roc_random_forest.png")
print(" - roc_knn_k5.png")
print(" - roc_decision_tree.png")
