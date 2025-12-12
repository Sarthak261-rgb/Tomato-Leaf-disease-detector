# per_class_report.py
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

# --- Config ---
DATA_DIR = Path("data_tomato_splits")
MODEL_PATH = "mobilenetv2_tomato.h5"
IMG_SIZE = (224, 224)
BATCH = 32

# --- Load test dataset (don't shuffle) ---
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="categorical",
    shuffle=False
)

class_names = test_ds.class_names

# --- Load model ---
model = tf.keras.models.load_model(MODEL_PATH)

# --- Collect y_true and y_pred ---
# y_true from dataset one-hot -> class indices
y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
y_true = np.argmax(y_true, axis=1)

# y_pred from model probabilities -> class indices
y_prob_list = []
for x, _ in test_ds:
    p = model.predict(x, verbose=0)
    y_prob_list.append(p)
y_prob = np.concatenate(y_prob_list, axis=0)
y_pred = np.argmax(y_prob, axis=1)

# --- Overall accuracy (sanity check) ---
acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Overall Test Accuracy: {acc*100:.2f}%")
print(f"✅ Misclassification Rate: {100-acc*100:.2f}%")

# --- Classification report (per class) ---
report_dict = classification_report(
    y_true, y_pred,
    target_names=class_names,
    output_dict=True, digits=4
)

# Convert to DataFrame (keep only per-class rows + averages)
df = pd.DataFrame(report_dict).T
# Reorder to put classes first, then averages
ordered_index = class_names + ["accuracy", "macro avg", "weighted avg"]
df = df.reindex(ordered_index)

# Pretty print
pd.set_option("display.max_columns", None)
print("\nPer-class metrics (precision, recall, f1-score, support):\n")
print(df[["precision", "recall", "f1-score", "support"]])

# Save CSV
out_csv = "per_class_report.csv"
df.to_csv(out_csv, float_format="%.4f")
print(f"\n💾 Saved: {out_csv}")
