import joblib
import numpy as np
from pathlib import Path
from glob import glob
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

# ---- Config ----
DATA_DIR = Path("data_tomato_splits")
IMG_SIZE = (224, 224)

# ---- Load dataset paths ----
def load_split(root):
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    paths, labels = [], []
    for i, c in enumerate(classes):
        for p in glob(str(root / c / "*")):
            if p.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(p); labels.append(i)
    return paths, np.array(labels), classes

def read_img(p):
    return np.array(Image.open(p).convert("RGB").resize(IMG_SIZE))

def extract(paths, base):
    feats = []
    for i in tqdm(range(0, len(paths), 64)):
        batch = np.stack([read_img(x) for x in paths[i:i+64]], axis=0)
        batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
        f = base.predict(batch, verbose=0)
        f = tf.keras.layers.GlobalAveragePooling2D()(tf.convert_to_tensor(f)).numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)

# ---- Load train/val/test paths ----
train_p, train_y, class_names = load_split(DATA_DIR / "train")
val_p, val_y, _               = load_split(DATA_DIR / "val")
test_p, test_y, _             = load_split(DATA_DIR / "test")

# ---- Feature extractor ----
base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,))
base.trainable = False

train_X = extract(train_p, base)
val_X   = extract(val_p, base)
test_X  = extract(test_p, base)

# ---- Scaling ----
scaler = StandardScaler()
train_Xs = scaler.fit_transform(train_X)
val_Xs   = scaler.transform(val_X)
test_Xs  = scaler.transform(test_X)

print("\nSaving ML models...")

# 1. SVM
svm = SVC(C=10, gamma="scale", probability=True)
svm.fit(np.vstack([train_Xs, val_Xs]), np.concatenate([train_y, val_y]))
joblib.dump(svm, "svm_model.pkl")

# 2. Random Forest
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(np.vstack([train_X, val_X]), np.concatenate([train_y, val_y]))
joblib.dump(rf, "rf_model.pkl")

# 3. KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(np.vstack([train_Xs, val_Xs]), np.concatenate([train_y, val_y]))
joblib.dump(knn, "knn_model.pkl")

# 4. Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(np.vstack([train_X, val_X]), np.concatenate([train_y, val_y]))
joblib.dump(dt, "dt_model.pkl")

# Save scaler too
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Models saved: svm_model.pkl, rf_model.pkl, knn_model.pkl, dt_model.pkl, scaler.pkl")
