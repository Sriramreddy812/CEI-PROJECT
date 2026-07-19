# 🛰️ Satellite Image Land-Use Classifier & Temporal Change Detector

A deep learning project that classifies satellite land-use imagery and detects temporal land-cover changes using **transfer learning**, **embedding similarity**, and an interactive **Streamlit dashboard**.

---

## 🚀 Live Demo

**Streamlit App:**  
https://cei-project-5wgbrbtcnaff4be8bl9nsh.streamlit.app/

---

## ✨ Features

- 🌍 Land-use classification using a fine-tuned **ResNet18**
- 🥇 Top-3 predictions with confidence scores
- 🔄 Temporal change detection using **512-dimensional image embeddings**
- 📊 ROC-based cosine similarity threshold selection
- 🎛️ **Bonus Task B:** Multi-threshold detection mode
  - High Recall
  - Balanced
  - High Precision
- 🔥 Pixel difference heatmap visualization
- 📈 Baseline CNN vs Transfer Learning comparison
- 🧪 Spatial leakage experiment
- 📉 Error analysis of top-5 misclassified class pairs

---

# 📂 Project Structure

```text
├── notebooks/
│   ├── 01_Data_Pipeline.ipynb
│   ├── 02_Baseline_CNN.ipynb
│   ├── 03_Transfer_Learning.ipynb
│   ├── 04_Temporal_Change_Detection.ipynb
│   └── 05_Evaluation.ipynb
│
├── app/
│   └── streamlit_app.py
│
├── models/
│   └── model.pt      (Generated after training - not included)
│
├── screenshots/
│   ├── predictions_changeDetection.png
│   ├── upload_and_predictions.png
│   └── heatmap.png
│
├── requirements.txt
└── README.md
```

# 📚 Datasets

## EuroSAT

- 27,000 RGB satellite images
- 10 land-use classes
- Downloaded automatically inside the notebooks using:

```python
torchvision.datasets.EuroSAT(download=True)
```

---

## UC Merced Land Use

- 2,100 RGB satellite images
- 21 land-use classes

Downloaded automatically during evaluation.

---

# 🚀 Running the Notebooks

The project is designed for **Google Colab**.

Run the notebooks sequentially:

### 1. Data Pipeline

```
01_Data_Pipeline.ipynb
```

- Dataset download
- Class visualization
- Spatial block split

---

### 2. Baseline CNN

```
02_Baseline_CNN.ipynb
```

- Scratch 3-layer CNN
- Training & validation curves
- Per-class F1 score

---

### 3. Transfer Learning

```
03_Transfer_Learning.ipynb
```

- Two-phase ResNet18 fine-tuning
- Frozen vs Unfrozen comparison
- UC Merced evaluation

At the end of this notebook, save

```
models/model.pt
```

---

### 4. Temporal Change Detection

```
04_Temporal_Change_Detection.ipynb
```

Includes:

- Feature embeddings
- Cosine similarity
- ROC curve
- Threshold selection
- Change heatmaps

---

### 5. Evaluation

```
05_Evaluation.ipynb
```

Includes:

- Spatial leakage experiment
- UC Merced evaluation
- Per-class metrics
- Error analysis

---

# 💻 Running the Dashboard Locally

Install dependencies

```bash
pip install -r requirements.txt
```

Place the trained model

```
models/model.pt
```

inside the **models** folder.

Run

```bash
streamlit run app/streamlit_app.py
```

The dashboard provides:

- Satellite image classification
- Confidence scores
- Top-3 predictions
- Cosine similarity
- Change detection
- Pixel difference heatmap
- Multi-threshold operating modes

---

# 🎁 Bonus Task Implemented

## ✅ Bonus B — Multi-threshold Toggle

The dashboard allows users to switch between three operating modes:

- High Recall
- Balanced
- High Precision

Different similarity thresholds allow users to observe how the change detection decision varies depending on the selected operating point.

---

# 📖 Project Approximations

## Spatial Block Split

The standard EuroSAT dataset does not contain geographic coordinates.

To approximate spatial separation, each class is partitioned into contiguous blocks instead of using a random shuffle, reducing spatial leakage.

---

## Temporal Change Simulation

EuroSAT does not contain real multi-temporal imagery.

Therefore, temporal image pairs are simulated using contiguous subsets of the dataset, with land-cover changes defined based on differing class assignments.

---

# 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab

---

# 📌 Repository Notes

The following files are **not included** in this repository due to GitHub size limitations:

- `data/`
- `assets/`
- `models/model.pt`

Datasets are downloaded automatically by running the notebooks.

---

## 📄 Project Report

👉 **[View the Project Report (PDF)](./Project_Report.pdf)**

The trained checkpoint (`models/model.pt`) is generated after completing **Notebook 3** and is automatically downloaded by the deployed Streamlit application from Google Drive if it is not present locally.
