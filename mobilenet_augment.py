
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

BASELINE = MODEL_DIR / "ham10000_mobilenetv2_finetuned.keras"
OUTPUT = MODEL_DIR / "ham10000_mobilenetv2_augmented.keras"
REPORT = RESULTS_DIR / "mobilenetv2_augmented_report.txt"

IMG_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 5
SEED = 42

tf.keras.utils.set_random_seed(SEED)

if not BASELINE.exists():
    raise FileNotFoundError(f"Missing model: {BASELINE}")

# Load metadata
metadata_path = RAW / "HAM10000_metadata"
if not metadata_path.exists():
    metadata_path = RAW / "HAM10000_metadata.csv"

df = pd.read_csv(metadata_path)

class_names = sorted(df["dx"].unique())
class_to_index = {name: i for i, name in enumerate(class_names)}
num_classes = len(class_names)

df["label"] = df["dx"].map(class_to_index).astype(np.int32)

# Find images
image_paths = {}

for folder_name in ["images_part_1", "images_part_2"]:
    for path in (RAW / folder_name).rglob("*.jpg"):
        if path.stem in image_paths:
            raise ValueError(f"Duplicate image ID: {path.stem}")
        image_paths[path.stem] = str(path)

df["image_path"] = df["image_id"].map(image_paths)

if df["image_path"].isna().any():
    raise ValueError("Some images are missing.")

# Recreate the lesion-level split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED,
)

train_idx, val_idx = next(
    splitter.split(df, groups=df["lesion_id"])
)

train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)

assert set(train_df["lesion_id"]).isdisjoint(
    set(val_df["lesion_id"])
)

# Class weights
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=train_df["label"].to_numpy(),
)

class_weights = {
    i: float(w) for i, w in enumerate(weights)
}

# Augment training images only
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10),
], name="image_augmentation")


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


def make_dataset(frame, training=False):
    ds = tf.data.Dataset.from_tensor_slices((
        frame["image_path"].to_numpy(),
        frame["label"].to_numpy(dtype=np.int32),
    ))

    if training:
        ds = ds.shuffle(
            len(frame),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        ds = ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df, training=False)

# Load best checkpoint
model = tf.keras.models.load_model(BASELINE)

# Find MobileNetV2 backbone
base_model = next(
    (
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model)
    ),
    None,
)

if base_model is None:
    raise ValueError("MobileNetV2 backbone not found.")

# Fine-tune only the final backbone layers
base_model.trainable = True

for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers[-30:]:
    if not isinstance(
        layer, tf.keras.layers.BatchNormalization
    ):
        layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
)

# Evaluate
y_true = val_df["label"].to_numpy()
probabilities = model.predict(val_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(y_true, y_pred)

macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    average="macro",
    zero_division=0,
)

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    target_names=class_names,
    zero_division=0,
)

print("\n=== AUGMENTATION RESULTS ===")
print(f"Validation accuracy: {accuracy:.4f}")
print(f"Validation macro F1: {macro_f1:.4f}")
print(report)

model.save(OUTPUT)

REPORT.write_text(
    f"Accuracy: {accuracy:.4f}\n"
    f"Macro F1: {macro_f1:.4f}\n"
    f"Class mapping: {class_to_index}\n\n{report}"
)

print("Model saved:", OUTPUT)
print("Report saved:", REPORT)
