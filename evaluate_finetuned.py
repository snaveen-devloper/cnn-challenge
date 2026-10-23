
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "models" / "ham10000_mobilenetv2_finetuned.keras"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (160, 160)
BATCH_SIZE = 16
SEED = 42

# Load metadata and reproduce the original split.
metadata_path = RAW / "HAM10000_metadata"
if not metadata_path.exists():
    metadata_path = RAW / "HAM10000_metadata.csv"

df = pd.read_csv(metadata_path)

class_names = sorted(df["dx"].unique())
class_to_index = {name: i for i, name in enumerate(class_names)}
df["label"] = df["dx"].map(class_to_index).astype(np.int32)

# Locate all images.
image_paths = {}
for folder_name in ["images_part_1", "images_part_2"]:
    folder = RAW / folder_name
    for path in folder.rglob("*.jpg"):
        if path.stem in image_paths:
            raise ValueError(f"Duplicate image ID: {path.stem}")
        image_paths[path.stem] = str(path)

df["image_path"] = df["image_id"].map(image_paths)
if df["image_path"].isna().any():
    raise ValueError("Some metadata images are missing.")

splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=SEED
)
train_idx, val_idx = next(
    splitter.split(df, groups=df["lesion_id"])
)
val_df = df.iloc[val_idx].reset_index(drop=True)

print("Validation images:", len(val_df))
print("Classes:", class_to_index)

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )
    return image, label

paths = val_df["image_path"].to_numpy()
labels = val_df["label"].to_numpy(dtype=np.int32)

val_ds = tf.data.Dataset.from_tensor_slices((paths, labels))
val_ds = val_ds.map(
    load_image, num_parallel_calls=tf.data.AUTOTUNE
)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Load only; do not retrain.
model = tf.keras.models.load_model(MODEL_PATH)

y_true = labels
probabilities = model.predict(val_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=np.arange(len(class_names)),
    average="macro",
    zero_division=0,
)

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    zero_division=0,
)

print(f"\nValidation accuracy: {accuracy:.4f}")
print(f"Validation macro F1: {macro_f1:.4f}")
print("\nClassification report:\n", report)

report_path = RESULTS / "finetuned_evaluation_report.txt"
report_path.write_text(
    f"Accuracy: {accuracy:.4f}\n"
    f"Macro F1: {macro_f1:.4f}\n\n{report}"
)

cm = confusion_matrix(
    y_true, y_pred, labels=np.arange(len(class_names))
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)
disp.plot(xticks_rotation=45, cmap="Blues", values_format="d")
plt.tight_layout()
plt.savefig(
    RESULTS / "finetuned_confusion_matrix.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close()

# Save the individual predictions for error analysis.
predictions = val_df[
    ["image_id", "lesion_id", "dx"]
].copy()
predictions["true_label"] = [
    class_names[i] for i in y_true
]
predictions["predicted_label"] = [
    class_names[i] for i in y_pred
]
predictions["confidence"] = probabilities.max(axis=1)
predictions["correct"] = y_true == y_pred

predictions.to_csv(
    RESULTS / "finetuned_predictions.csv", index=False
)

print("\nSaved report:", report_path)
print("Saved confusion matrix:",
      RESULTS / "finetuned_confusion_matrix.png")
print("Saved predictions:",
      RESULTS / "finetuned_predictions.csv")
