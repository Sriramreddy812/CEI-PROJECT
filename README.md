# Satellite Image Land-Use Classifier & Temporal Change Detector

A computer vision project that classifies land-use types from satellite
imagery and detects land-cover changes between two time periods, using
transfer learning and embedding-based change detection.

## Project Structure

```
notebooks/
    01_Data_Pipeline.ipynb              - load EuroSAT, spatial block split
    02_Baseline_CNN.ipynb               - scratch 3-layer CNN baseline
    03_Transfer_Learning.ipynb          - two-phase ResNet18 fine-tuning + UC Merced eval
    04_Temporal_Change_Detection.ipynb  - embeddings, cosine similarity, ROC curve
    05_Evaluation.ipynb                 - spatial leakage experiment, error analysis

app/
    streamlit_app.py                    - geo-dashboard (classify + change detection)

requirements.txt
README.md
```

## Datasets

- **EuroSAT** (27,000 images, 10 classes) — downloaded automatically inside
  the notebooks via `torchvision.datasets.EuroSAT(download=True)`.
- **UC Merced Land Use** (2,100 images, 21 classes) — downloaded
  automatically from the official host inside Notebooks 3 and 5.

## How to Run the Notebooks

All notebooks are written for Google Colab: open each one in Colab and use
**Runtime → Run all**. No local files or paths are required — datasets
download automatically.

Run them in order:
1. `01_Data_Pipeline.ipynb`
2. `02_Baseline_CNN.ipynb`
3. `03_Transfer_Learning.ipynb` — saves `resnet18_landuse.pt` at the end.
   **Download this file** (or save it to Google Drive) before your Colab
   runtime disconnects, since later notebooks and the dashboard need it.
4. `04_Temporal_Change_Detection.ipynb` — needs `resnet18_landuse.pt`
   uploaded at the start if running in a fresh Colab session.
5. `05_Evaluation.ipynb` — also needs `resnet18_landuse.pt` uploaded.

## How to Run the Dashboard

The dashboard runs locally (not in Colab):

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Place your `resnet18_landuse.pt` checkpoint (downloaded from Notebook 3)
   in the `app/` folder, next to `streamlit_app.py`.
3. Run:
   ```
   streamlit run app/streamlit_app.py
   ```
4. Upload a "before" and "after" tile to see the predicted land-use class
   for each, their embedding similarity, and a change heatmap.

No internet connection is needed to run the dashboard itself once the
checkpoint file is in place.

## Notes on Approximations

Two things in this project are approximated because the underlying data
doesn't provide what the brief asks for directly:

- **Spatial block split**: standard EuroSAT has no geographic coordinates,
  so the "spatial block" split groups each class's images into contiguous
  chunks (assumed to approximate nearby tiles) instead of shuffling
  randomly.
- **T1/T2 change simulation**: EuroSAT has no real before/after imagery, so
  Notebook 4 simulates temporal pairs from contiguous per-class chunks, and
  defines "changed" as a class change between the two tiles in a pair.

Both approximations are explained in detail in their respective notebooks.
