# confusion_matrix_cnn_pretty.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

# --- Config ---
DATA_DIR = Path("data_tomato_splits")
MODEL_PATH = "mobilenetv2_tomato.h5"
IMG_SIZE = (224, 224)
BATCH = 32

# --- Load dataset (test only) ---
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="categorical",
    shuffle=False
)

class_names = test_ds.class_names
model = tf.keras.models.load_model(MODEL_PATH)

# --- Get true labels and predictions ---
y_true = np.concatenate([y.numpy() for x, y in test_ds])
y_true = np.argmax(y_true, axis=1)

y_pred = []
for x, _ in test_ds:
    preds = model.predict(x, verbose=0)
    y_pred.append(np.argmax(preds, axis=1))
y_pred = np.concatenate(y_pred)

# --- Compute confusion matrix ---
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalized (0–1 scale)

# --- Plot settings ---
plt.figure(figsize=(10, 8))
sns.set(font_scale=1.0)
sns.set_style("whitegrid")

ax = sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={'label': 'Proportion'}
)

plt.title("Confusion Matrix — MobileNetV2 Tomato Disease Detection", fontsize=14, weight='bold', pad=20)
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("True Class", fontsize=12)
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("cnn_confusion_matrix_pretty.png", dpi=300, bbox_inches="tight")
plt.show()

# --- Optional: Print text summary (good for report appendix) ---
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
print("✅ Saved: cnn_confusion_matrix_pretty.png")
