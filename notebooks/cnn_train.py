# cnn_train.py  (fixed)
import tensorflow as tf
from pathlib import Path

# --- Config ---
DATA_DIR = Path("data_tomato_splits")   # make sure split_dataset.py already created this
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 15
SEED = 42

# --- Load datasets ---
train = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="categorical",
    shuffle=True,
    seed=SEED,
)

val = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR / "val",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="categorical",
    shuffle=False,
)

test = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="categorical",
    shuffle=False,
)

# ✅ Capture class names BEFORE prefetch (PrefetchDataset has no .class_names)
class_names = train.class_names
num_classes = len(class_names)
print("Detected classes:", class_names)
print("Num classes:", num_classes)

# --- Performance tweaks ---
AUTOTUNE = tf.data.AUTOTUNE
train = train.prefetch(AUTOTUNE)
val = val.prefetch(AUTOTUNE)
test = test.prefetch(AUTOTUNE)

# --- Build model (MobileNetV2 transfer learning) ---
base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base.trainable = False  # freeze base for first stage

data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
], name="augmentation")

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = data_aug(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)  # ✅ use num_classes
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# --- Train ---
history = model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS
)

# --- Evaluate ---
loss, acc = model.evaluate(test)
print(f"✅ Test Accuracy: {acc*100:.2f}%")
print(f"✅ Misclassification Rate: {100 - acc*100:.2f}%")

# --- Save model ---
model.save("mobilenetv2_tomato.h5")
print("✅ Saved model: mobilenetv2_tomato.h5")
