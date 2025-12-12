# evaluate_saved_model.py
import tensorflow as tf
from pathlib import Path

DATA_DIR = Path("data_tomato_splits")
IMG_SIZE = (224, 224)
BATCH = 32

# Load test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="categorical"
)

# Load your trained model
model = tf.keras.models.load_model("mobilenetv2_tomato.h5")

# Evaluate
loss, acc = model.evaluate(test_ds)
print(f"✅ Test Accuracy: {acc*100:.2f}%")
print(f"✅ Misclassification Rate: {100-acc*100:.2f}%")
print(f"✅ Test Loss: {loss:.4f}")