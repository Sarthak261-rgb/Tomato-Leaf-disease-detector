import numpy as np
from pathlib import Path
from glob import glob
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

DATA_DIR = Path("data_tomato_splits")
IMG_SIZE = (224, 224)

def load(root):
    classes = sorted([d.name for d in (root).iterdir() if d.is_dir()])
    paths, labels = [], []
    for i, c in enumerate(classes):
        for p in glob(str(root / c / "*")):
            if Path(p).suffix.lower() in [".jpg", ".jpeg", ".png"]:
                paths.append(p); labels.append(i)
    return paths, np.array(labels), classes

def read_img(p):
    return np.array(Image.open(p).convert("RGB").resize(IMG_SIZE), dtype=np.float32)

def extract(paths, base):
    feats = []
    for i in tqdm(range(0, len(paths), 64)):
        batch = np.stack([read_img(p) for p in paths[i:i+64]], 0)
        batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
        f = base.predict(batch, verbose=0)
        f = tf.keras.layers.GlobalAveragePooling2D()(tf.convert_to_tensor(f)).numpy()
        feats.append(f)
    return np.concatenate(feats, 0)

base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=IMG_SIZE+(3,))
tr_p, tr_y, _ = load(DATA_DIR / "train")
va_p, va_y, _ = load(DATA_DIR / "val")
te_p, te_y, classes = load(DATA_DIR / "test")

tr_X = extract(tr_p, base)
va_X = extract(va_p, base)
te_X = extract(te_p, base)

sc = StandardScaler()
tr_Xs = sc.fit_transform(tr_X)
va_Xs = sc.transform(va_X)
te_Xs = sc.transform(te_X)

def run(name, clf, Xtr, ytr, Xva, yva, Xte, yte):
    clf.fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    mis = 1 - acc
    print(f"{name:>12} | Accuracy: {acc*100:.2f}% | Misclassification: {mis*100:.2f}%")

run("SVM (RBF)", SVC(kernel="rbf", C=10, gamma="scale"), tr_Xs, tr_y, va_Xs, va_y, te_Xs, te_y)
run("RandomForest", RandomForestClassifier(n_estimators=300, random_state=42),
    tr_X, tr_y, va_X, va_y, te_X, te_y)
run("KNN (k=5)", KNeighborsClassifier(n_neighbors=5), tr_Xs, tr_y, va_Xs, va_y, te_Xs, te_y)
run("DecisionTree", DecisionTreeClassifier(random_state=42), tr_X, tr_y, va_X, va_y, te_X, te_y)
print("✅ Baseline evaluation complete.")