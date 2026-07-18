# 🛰️ Satellite Image Land-Use Classifier & Temporal Change Detector

A deep learning project that classifies satellite land-use images and detects temporal land-cover changes using **transfer learning**, **embedding similarity**, and an interactive **Streamlit dashboard**.

---

## ✨ Features

- 🌍 Land-use classification using a fine-tuned **ResNet18**
- 🔄 Temporal change detection using **512-dimensional embeddings**
- 📊 ROC-based cosine similarity threshold selection
- 🔥 Pixel difference heatmap visualization
- 📈 Baseline CNN vs Transfer Learning comparison
- 🧪 Spatial leakage experiment
- 📉 Error analysis on top-5 misclassified pairs
- 🎛️ **Bonus Task B:** Multi-threshold detection mode (High Recall / Balanced / High Precision)

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
├── requirements.txt
└── README.md
```

---

# 📚 Datasets

### EuroSAT
- 27,000 RGB satellite images
- 10 land-use classes
- Automatically downloaded inside the notebooks using:

```python
torchvision.datasets.EuroSAT(download=True)
```

---

### UC Merced Land Use

- 2,100 RGB images
- 21 land-use classes
- Downloaded automatically in the evaluation notebooks.

---

# 🚀 Running the Notebooks

All notebooks are designed for **Google Colab**.

Run them **in order**:

### 1️⃣ Data Pipeline

```
01_Data_Pipeline.ipynb
```

- Dataset download
- Class visualization
- Spatial block split

---

### 2️⃣ Baseline CNN

```
02_Baseline_CNN.ipynb
```

- Scratch 3-layer CNN
- Loss curves
- Per-class F1

---

### 3️⃣ Transfer Learning

```
03_Transfer_Learning.ipynb
```

- Two-phase ResNet18 fine-tuning
- Frozen vs unfrozen comparison
- UC Merced evaluation

At the end of this notebook, save:

```
model.pt
```

This checkpoint is required for the dashboard and later notebooks.

---

### 4️⃣ Temporal Change Detection

```
04_Temporal_Change_Detection.ipynb
```

- Embedding extraction
- Cosine similarity
- ROC curve
- Threshold selection
- Change heatmaps

---

### 5️⃣ Evaluation

```
05_Evaluation.ipynb
```

Includes:

- Spatial leakage experiment
- UC Merced evaluation
- Per-class metrics
- Error analysis

---

# 💻 Running the Streamlit Dashboard

Install dependencies

```bash
pip install -r requirements.txt
```

Place the trained checkpoint

```
models/model.pt
```

inside the project.

Launch the application

```bash
streamlit run streamlit_app.py
```

The dashboard provides:

- Satellite image classification
- Confidence scores
- Top-3 predictions
- Cosine similarity
- Change detection
- Pixel difference heatmap
- **Bonus:** Multi-threshold detection mode

---

# 🎁 Bonus Task Implemented

## ✅ Bonus B — Multi-threshold Toggle

The dashboard allows switching between three operating modes:

- High Recall
- Balanced
- High Precision

Users can instantly observe how different cosine similarity thresholds affect change detection results.

---

# 📌 Notes

Since the datasets are large, the following folders/files are **not included** in this repository:

- `data/`
- `assets/`
- `models/model.pt`

Please download the datasets by running the notebooks.

Place the trained checkpoint (`model.pt`) inside the `models/` folder before launching the Streamlit dashboard.

---

# 📖 Project Approximations

### Spatial Block Split

EuroSAT does not provide geographic coordinates.

To approximate spatial separation, each class is divided into contiguous blocks rather than random shuffling, reducing potential spatial leakage.

---

### Temporal Change Simulation

EuroSAT does not contain true multi-temporal imagery.

Notebook 4 therefore simulates T1/T2 image pairs by partitioning each class into contiguous subsets and defining changes using paired class assignments.

These approximations are fully documented inside the notebooks.

---

# 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab

---

# 📌 Repository Note

Large datasets and trained model checkpoints are intentionally excluded from the repository due to GitHub file size limits.

Running the notebooks automatically downloads the required datasets, while the trained model checkpoint should be generated from **Notebook 3** or placed manually inside the `models/` directory.
