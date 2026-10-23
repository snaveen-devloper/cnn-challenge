# 🔬 HAM10000 Skin Lesion Classification — CNN Challenge

> **PS 15 | Naveen S | RA2311026050208 | Section A**  
> Deep Learning Image Classification Challenge

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![Dataset](https://img.shields.io/badge/Dataset-HAM10000-green)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

---

## 📋 Problem Statement

Seven-class dermoscopic skin-lesion classification on the **HAM10000** dataset (10,015 images). The dataset is heavily imbalanced — *nv* (melanocytic naevi) makes up 67% of images — so **macro-averaged F1-score** is the primary evaluation metric.

| Class | Full Name | Images |
|-------|-----------|--------|
| `nv`    | Melanocytic Naevi | 6,705 |
| `mel`   | Melanoma | 1,113 |
| `bkl`   | Benign Keratosis | 1,099 |
| `bcc`   | Basal Cell Carcinoma | 514 |
| `akiec` | Actinic Keratoses | 327 |
| `vasc`  | Vascular Lesions | 142 |
| `df`    | Dermatofibroma | 115 |

---

## 🗂️ Project Structure

```
cnn-challenge/
├── cnn_train.py             # Model A — Custom CNN from scratch
├── mobilenet_train.py       # Model B — MobileNetV2 (frozen backbone)
├── mobilenet_finetune.py    # Model C — MobileNetV2 (fine-tuned top 30 layers)
├── mobilenet_augment.py     # Model D — Fine-tuned + data augmentation
├── evaluate_finetuned.py    # Standalone evaluation + confusion matrix
├── app.py                   # Streamlit inference demo
├── data/                    # Raw images & metadata (not tracked in git)
├── models/                  # Saved .keras models (not tracked in git)
├── results/                 # Plots, reports, predictions (not tracked in git)
└── README.md
```

> **Note:** `data/`, `models/`, and `results/` are excluded from git (see `.gitignore`) due to size (~5.6 GB total). Download the dataset separately — see [Setup](#️-setup).

---

## 🏗️ Model Progression

| # | Model | Description |
|---|-------|-------------|
| **A** | Custom CNN | 3× Conv-MaxPool (32→64→128), GAP, Dropout(0.3), Dense(7) — trained from scratch |
| **B** | MobileNetV2 (frozen) | ImageNet backbone fully frozen; only the Dense head is trained |
| **C** | MobileNetV2 (fine-tuned) | Last 30 backbone layers unfrozen (BN frozen); LR = 1e-5 |
| **D** | MobileNetV2 (aug. fine-tuned) | Same as C + random flip, ±8° rotation, ±10% zoom on training set |

**Key design choices:**
- **Lesion-level split** via `GroupShuffleSplit` (seed 42) — prevents data leakage where multiple crops of the same lesion end up in both train and val
- **Balanced class weights** via scikit-learn to handle severe class imbalance
- **Early stopping** (patience = 2, restore best weights) on validation loss

---

## 📊 Results

All metrics measured on the **validation set** (2,024 images, 20% lesion-level hold-out).

| Model | Val. Accuracy | Macro F1 | F1 – mel | F1 – nv | F1 – df |
|-------|:---:|:---:|:---:|:---:|:---:|
| A – Custom CNN (scratch)            | 0.47 | 0.30 | 0.38 | 0.68 | 0.09 |
| B – MobileNetV2 (frozen)            | 0.65 | 0.47 | 0.46 | 0.79 | 0.18 |
| **C – MobileNetV2 (fine-tuned)**    | **0.72** | **0.54** | **0.49** | **0.86** | **0.17** |
| D – MobileNetV2 (aug. fine-tuned)   | 0.63 | 0.44 | 0.43 | 0.78 | 0.25 |

> **Best model:** Model C — MobileNetV2 fine-tuned (last 30 layers), LR = 1e-5, balanced class weights.

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/snaveen-devloper/cnn-challenge.git
cd cnn-challenge
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install tensorflow scikit-learn pandas matplotlib streamlit pillow
```

### 4. Download the dataset
Get HAM10000 from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) and place it as:

```
data/
└── raw/
    ├── HAM10000_metadata.csv
    ├── images_part_1/   (*.jpg)
    └── images_part_2/   (*.jpg)
```

---

## 🚀 Running

Run the scripts in order (each builds on the previous model checkpoint):

```bash
# Step 1 — Train custom CNN baseline
python cnn_train.py

# Step 2 — Train MobileNetV2 with frozen backbone
python mobilenet_train.py

# Step 3 — Fine-tune the top 30 backbone layers
python mobilenet_finetune.py

# Step 4 — Fine-tune with data augmentation
python mobilenet_augment.py

# (Optional) Re-evaluate the fine-tuned model & generate confusion matrix
python evaluate_finetuned.py
```

Each script saves its model to `models/` and its classification report + figures to `results/`.

---

## 🖥️ Streamlit Demo

```bash
streamlit run app.py
```

Upload any dermoscopic image and the app returns the predicted class and a per-class probability bar chart.

> ⚠️ **Disclaimer:** This is an educational research prototype only. It is **not** a medical diagnostic tool and must not be used for clinical decisions.

---

## 📁 Outputs

| File | Description |
|------|-------------|
| `results/classification_report.txt` | CNN (Model A) per-class precision / recall / F1 |
| `results/mobilenetv2_report.txt` | Model B metrics |
| `results/mobilenetv2_finetuned_report.txt` | Model C metrics |
| `results/mobilenetv2_augmented_report.txt` | Model D metrics |
| `results/confusion_matrix.png` | CNN confusion matrix |
| `results/finetuned_confusion_matrix.png` | Model C confusion matrix |
| `results/training_history.png` | CNN accuracy learning curve |
| `results/finetuned_predictions.csv` | Per-image predictions with confidence scores |

---

## 🔮 Future Work

- Replace cross-entropy with **Focal Loss** (γ = 2) to boost minority-class recall
- **Grad-CAM** visualisations to verify the model attends to lesion regions
- Test generalisation on ISIC 2019 / external datasets
- Address demographic bias with diverse skin-tone data augmentation

---

## 📄 Dataset Citation

```
Tschandl, P., Rosendahl, C., & Kittler, H. (2018).
The HAM10000 dataset, a large collection of multi-source dermatoscopic images
of common pigmented skin lesions.
Scientific Data, 5, 180161.
https://doi.org/10.1038/sdata.2018.161
```
