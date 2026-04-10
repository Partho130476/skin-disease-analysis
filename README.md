# 🩺 Skin Disease Analysis

A deep learning-based skin disease classification system that identifies **23 skin conditions** from dermoscopic images. Built with EfficientNetB0 and served via a Flask web application.

---

## 📌 Overview

This project uses a convolutional neural network fine-tuned on the [ISIC / DermNet skin disease dataset](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) to classify uploaded skin images into one of 23 disease categories, returning the top-5 predictions with confidence scores.

---

## 🧠 Supported Disease Classes

| # | Class |
|---|-------|
| 1 | Acne and Rosacea |
| 2 | Actinic Keratosis / Basal Cell Carcinoma |
| 3 | Atopic Dermatitis |
| 4 | Bullous Disease |
| 5 | Cellulitis / Impetigo / Bacterial Infections |
| 6 | Eczema |
| 7 | Exanthems and Drug Eruptions |
| 8 | Hair Loss / Alopecia |
| 9 | Herpes / HPV / STDs |
| 10 | Light Diseases / Pigmentation Disorders |
| 11 | Lupus / Connective Tissue Diseases |
| 12 | Melanoma / Nevi / Moles |
| 13 | Nail Fungus / Nail Disease |
| 14 | Poison Ivy / Contact Dermatitis |
| 15 | Psoriasis / Lichen Planus |
| 16 | Scabies / Lyme Disease / Infestations |
| 17 | Seborrheic Keratoses / Benign Tumors |
| 18 | Systemic Disease |
| 19 | Tinea / Ringworm / Candidiasis / Fungal Infections |
| 20 | Urticaria / Hives |
| 21 | Vascular Tumors |
| 22 | Vasculitis |
| 23 | Warts / Molluscum / Viral Infections |

---

## 🏗️ Architecture

- **Model**: EfficientNetB0 (transfer learning, fine-tuned)
- **Input size**: 260×260 px
- **Backend**: Flask (Python)
- **Frontend**: HTML/CSS + Chart.js (probability bar chart)
- **Model storage**: Serialized via `joblib` as architecture + weights dict (`.pkl`)

```
skin-disease-analysis/
├── app.py                        # Flask application
├── skin_disease_analysis.py      # Training script (Google Colab)
├── skin_disease_analysis.ipynb   # Training notebook
├── templates/
│   └── index.html                # Web UI
├── static/
│   └── uploads/                  # Uploaded images (runtime)
├── model/
│   └── model.pkl                 # ← Place trained model here (see below)
├── style.css
├── package.json                  # Chart.js dependency
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/<Partho130476>/skin-disease-analysis.git
cd skin-disease-analysis
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the trained model

The trained model file (`model.pkl`) is **not included** in this repository due to file size constraints.

- **Option A**: Train it yourself using `skin_disease_analysis.ipynb` on Google Colab.
- **Option B**: Download the pre-trained model from [Google Drive / Releases](#) *(link your model here)* and place it at `model/model.pkl`.

### 4. Run the Flask server

```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000`.

---

## 🖥️ Usage

1. Navigate to the web UI.
2. Upload a skin image (`.jpg`, `.jpeg`, `.png`, `.gif`).
3. The model returns:
   - **Predicted class** with confidence %
   - **Top-5 predictions** visualized as a bar chart

> ⚠️ **Disclaimer**: This tool is for educational and research purposes only. It is **not** a substitute for professional medical diagnosis.

---

## 🧪 Training

The model was trained on the DermNet dataset using Google Colab. See `skin_disease_analysis.ipynb` for the full pipeline:

- Data loading & train/val/test split
- EfficientNetB0 fine-tuning with `ImageDataGenerator`
- Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`
- Model export via `joblib`

---

## 📦 Requirements

See [`requirements.txt`](requirements.txt).

---

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

## 🙋 Author

**Partho** — B.Sc. CSE, North East University Bangladesh  
Feel free to open an issue or submit a pull request.
