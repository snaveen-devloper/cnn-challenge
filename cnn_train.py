
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# ==================================================
# 2. LOAD METADATA
# ==================================================

metadata_path = RAW / "HAM10000_metadata"

if not metadata_path.exists():
    metadata_path = RAW / "HAM10000_metadata.csv"

if not metadata_path.exists():
    raise FileNotFoundError(
        f"Metadata file not found in {RAW}"
    )

df = pd.read_csv(metadata_path)

required_columns = {"image_id", "lesion_id", "dx"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Missing required columns: "
        f"{required_columns - set(df.columns)}"
    )

if df["dx"].isna().any() or df["lesion_id"].isna().any():
    raise ValueError("Missing diagnosis or lesion IDs detected.")

# ==================================================
# 3. FIND IMAGE FILES
# ==================================================

image_paths = {}

for folder_name in ["images_part_1", "images_part_2"]:
    folder = RAW / folder_name

    if not folder.exists():
        raise FileNotFoundError(
            f"Image folder not found: {folder}"
        )

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".jpg":
            image_id = path.stem

            if image_id in image_paths:
                raise ValueError(
                    f"Duplicate image ID: {image_id}"
                )

            image_paths[image_id] = str(path)

df["image_path"] = df["image_id"].map(image_paths)

missing = int(df["image_path"].isna().sum())

print("Metadata rows:", len(df))
print("Unique image files found:", len(image_paths))
print("Images missing from metadata:", missing)

if missing:
    missing_ids = df.loc[
        df["image_path"].isna(), "image_id"
    ].head(10).tolist()

    raise ValueError(
        f"Images are missing. Example IDs: {missing_ids}"
    )

if df["image_id"].duplicated().any():
    raise ValueError("Duplicate image IDs in metadata.")

# ==================================================
# 4. ENCODE THE SEVEN CLASSES
# ==================================================

class_names = sorted(df["dx"].unique().tolist())
num_classes = len(class_names)

if num_classes != 7:
    raise ValueError(
        f"Expected 7 classes, but found {num_classes}: "
        f"{class_names}"
    )

class_to_index = {
    name: index
    for index, name in enumerate(class_names)
}

df["label"] = (
    df["dx"].map(class_to_index).astype(np.int32)
)

print("\nClass mapping:", class_to_index)
print("\nClass counts:")
print(df["dx"].value_counts().reindex(class_names))

# ==================================================
# 5. LESION-LEVEL TRAIN/VALIDATION SPLIT
# ==================================================
# Images from the same lesion must never appear in
# both training and validation sets.

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

train_lesions = set(train_df["lesion_id"])
val_lesions = set(val_df["lesion_id"])

assert train_lesions.isdisjoint(val_lesions)

print("\nTraining images:", len(train_df))
print("Validation images:", len(val_df))
print("Training lesions:", len(train_lesions))
print("Validation lesions:", len(val_lesions))
print("Lesion overlap:", len(train_lesions & val_lesions))

# Ensure every class is represented in both partitions.
for class_name in class_names:
    if not (train_df["dx"] == class_name).any():
        raise ValueError(
            f"Class {class_name} is missing from training data."
        )

    if not (val_df["dx"] == class_name).any():
        raise ValueError(
            f"Class {class_name} is missing from validation data."
        )

# ==================================================
# 6. CALCULATE CLASS WEIGHTS
# ==================================================
# IMPORTANT: num_classes is defined above.
# Rare classes receive greater training weight.

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=train_df["label"].to_numpy(),
)

class_weights = {
    int(index): float(weight)
    for index, weight in enumerate(weights)
}

print("\nClass weights:")
for class_name, index in class_to_index.items():
    print(f"{class_name}: {class_weights[index]:.4f}")

# ==================================================
# 7. IMAGE LOADING PIPELINE
# ==================================================

def load_image(path, label):
    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_jpeg(
        image_bytes,
        channels=3,
    )

    image = tf.image.resize(
        image,
        IMG_SIZE,
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def make_dataset(frame, training=False):
    paths = frame["image_path"].to_numpy()
    labels = frame["label"].to_numpy(dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(frame),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


train_ds = make_dataset(
    train_df,
    training=True,
)

val_ds = make_dataset(
    val_df,
    training=False,
)

# ==================================================
# 8. BUILD THE CNN
# ==================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    ),

    tf.keras.layers.Conv2D(
        32, 3, padding="same", activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64, 3, padding="same", activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        128, 3, padding="same", activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        64, activation="relu"
    ),

    tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
    ),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ==================================================
# 9. TRAIN WITH CLASS WEIGHTS
# ==================================================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
)

# ==================================================
# 10. EVALUATE THE MODEL
# ==================================================

y_true = val_df["label"].to_numpy()

probabilities = model.predict(
    val_ds,
    verbose=1,
)

y_pred = np.argmax(
    probabilities,
    axis=1,
)

accuracy = accuracy_score(
    y_true,
    y_pred,
)

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

print("\n========== FINAL RESULTS ==========")
print(f"Validation accuracy: {accuracy:.4f}")
print(f"Validation macro F1: {macro_f1:.4f}")
print("\nClassification report:\n")
print(report)

report_path = RESULTS_DIR / "classification_report.txt"

report_path.write_text(
    report
    + f"\nValidation accuracy: {accuracy:.4f}"
    + f"\nValidation macro F1: {macro_f1:.4f}"
    + f"\nClass mapping: {class_to_index}\n"
)

# ==================================================
# 11. SAVE CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

disp.plot(
    xticks_rotation=45,
    cmap="Blues",
)

plt.title("HAM10000 Confusion Matrix")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=150,
)

plt.close()

# ==================================================
# 12. SAVE TRAINING HISTORY
# ==================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training accuracy",
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation accuracy",
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training History")
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "training_history.png",
    dpi=150,
)

plt.close()

# ==================================================
# 13. SAVE MODEL
# ==================================================

model_path = MODEL_DIR / "ham10000_cnn.keras"

model.save(model_path)

print("\nModel saved to:", model_path)
print("Classification report:", report_path)
print("Confusion matrix:", RESULTS_DIR / "confusion_matrix.png")
print("Training history:", RESULTS_DIR / "training_history.png")
print("Class mapping:", class_to_index)
print("\nTraining and evaluation completed.")
