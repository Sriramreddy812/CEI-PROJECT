"""
Geo-Dashboard — Satellite Image Land-Use Classifier & Temporal Change Detector

Run locally with:
    streamlit run app/streamlit_app.py

Requires the fine-tuned checkpoint (model.pt) from Notebook 3 to be
placed next to this script, or update CHECKPOINT_PATH below. Once the
checkpoint is in place, the app needs no internet connection to run.
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

# EuroSAT's 10 classes, in the same order used during training
CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

CHECKPOINT_PATH = "models/model.pt"

# Similarity threshold chosen from the ROC curve in Notebook 4.
# Update this value to match whatever threshold your own Notebook 4 run picked.
THRESHOLDS = {
    "High Recall": 0.35,
    "Balanced": 0.464,
    "High Precision": 0.60,
}

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


@st.cache_resource
def load_models():
    # Classifier model: outputs class scores for the 10 EuroSAT classes
    classifier_model = models.resnet18(weights=None)
    classifier_model.fc = nn.Linear(classifier_model.fc.in_features, len(CLASS_NAMES))
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    classifier_model.load_state_dict(state_dict)
    classifier_model.eval()

    # Embedding model: same weights, but classifier head removed -> 512-dim output
    embedding_model = models.resnet18(weights=None)
    embedding_model.fc = nn.Linear(embedding_model.fc.in_features, len(CLASS_NAMES))
    embedding_model.load_state_dict(state_dict)
    embedding_model.fc = nn.Identity()
    embedding_model.eval()

    return classifier_model, embedding_model


def predict(image_tensor, classifier_model):
    with torch.no_grad():
        logits = classifier_model(image_tensor.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)[0]

        predicted_idx = torch.argmax(probs).item()
        confidence = probs[predicted_idx].item()

        top3_probs, top3_idx = torch.topk(probs, 3)

    return (
        CLASS_NAMES[predicted_idx],
        confidence,
        top3_idx.tolist(),
        top3_probs.tolist(),
    )


def get_embedding(image_tensor, embedding_model):
    with torch.no_grad():
        embedding = embedding_model(image_tensor.unsqueeze(0))[0]
    return embedding.numpy()


def unnormalize(image_tensor):
    img = image_tensor.numpy().transpose(1, 2, 0)
    img = (img * 0.5) + 0.5
    return np.clip(img, 0, 1)


st.set_page_config(page_title="Land-Use & Change Detector", layout="wide")
st.sidebar.title("About")

st.sidebar.info(
    """
This application uses a fine-tuned ResNet18 model trained on the EuroSAT
dataset to:

• Classify satellite land-use images
• Detect changes between two satellite images using embedding similarity
"""
)
threshold_mode = st.sidebar.radio(
    "Detection Mode",
    ["High Recall", "Balanced", "High Precision"]
)

SIMILARITY_THRESHOLD = THRESHOLDS[threshold_mode]
st.title("🛰️ Satellite Land-Use Classifier & Temporal Change Detector")
st.write("Upload two tiles of the same area (before / after) to classify each and check for land-use change.")

classifier_model, embedding_model = load_models()

col1, col2 = st.columns(2)
with col1:
    t1_file = st.file_uploader("Before (T1) tile", type=["jpg", "jpeg", "png", "tif", "tiff"])
with col2:
    t2_file = st.file_uploader("After (T2) tile", type=["jpg", "jpeg", "png", "tif", "tiff"])

if t1_file is not None and t2_file is not None:
    t1_image = Image.open(t1_file).convert("RGB")
    t2_image = Image.open(t2_file).convert("RGB")

    t1_tensor = transform(t1_image)
    t2_tensor = transform(t2_image)

    # --- classification for each tile ---
    t1_class, t1_confidence, t1_top3_idx, t1_top3_probs = predict(t1_tensor,classifier_model,)
    t2_class, t2_confidence, t2_top3_idx, t2_top3_probs = predict(t2_tensor,classifier_model,)

    col1, col2 = st.columns(2)
    with col1:
        st.image(t1_image, caption="T1 (before)", use_container_width=True)
        st.write(f"**Predicted class:** {t1_class}")
        st.write(f"**Confidence:** {t1_confidence:.2%}")
        st.write("**Top-3 Predictions**")

        for idx, prob in zip(t1_top3_idx, t1_top3_probs):
            st.write(f"- {CLASS_NAMES[idx]} ({prob:.2%})")
    with col2:
        st.image(t2_image, caption="T2 (after)", use_container_width=True)
        st.write(f"**Predicted class:** {t2_class}")
        st.write(f"**Confidence:** {t2_confidence:.2%}")
        st.write("**Top-3 Predictions**")

        for idx, prob in zip(t2_top3_idx, t2_top3_probs):
            st.write(f"- {CLASS_NAMES[idx]} ({prob:.2%})")

    # --- cosine similarity between embeddings ---
    t1_embedding = get_embedding(t1_tensor, embedding_model)
    t2_embedding = get_embedding(t2_tensor, embedding_model)
    cosine_similarity = float(
        np.dot(t1_embedding, t2_embedding)
        / (np.linalg.norm(t1_embedding) * np.linalg.norm(t2_embedding))
    )

    st.subheader("Change Detection")
    st.write(f"**Cosine similarity:** {cosine_similarity:.3f}")
    st.write(f"**Detection Mode:** {threshold_mode}")
    st.write(f"**Similarity Threshold:** {SIMILARITY_THRESHOLD:.3f}")

    if cosine_similarity < SIMILARITY_THRESHOLD:
        st.error("🔴 Change Detected")
    else:
        st.success("🟢 No Significant Change")

    # --- side-by-side heatmap ---
    t1_display = unnormalize(t1_tensor)
    t2_display = unnormalize(t2_tensor)
    diff_heatmap = np.mean(np.abs(t1_display - t2_display), axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(t1_display)
    axes[0].set_title("T1")
    axes[0].axis("off")

    axes[1].imshow(t2_display)
    axes[1].set_title("T2")
    axes[1].axis("off")

    axes[2].imshow(diff_heatmap, cmap="hot")
    axes[2].set_title("Pixel Difference Heatmap")
    axes[2].axis("off")

    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Upload both a T1 and a T2 tile to see results.")
st.markdown("---")
st.caption(
    "Built using Streamlit • PyTorch • ResNet18 • EuroSAT Dataset"
)
