import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

# 1. Path Setup
MODEL_PATH = '/content/models/milestone3_final_cnn.h5'

# 2. Final Mapping
SCANNER_NAMES = {
    0: "Canon CanoScan LiDE 120",
    1: "Epson Perfection V39"
}

@st.cache_resource
def load_my_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_my_model()

st.title("🔍 Digital Forensic Scanner Analysis")

if model is None:
    st.error(f"❌ Model missing at {MODEL_PATH}")
else:
    st.sidebar.success("✅ Model Loaded")

uploaded_file = st.file_uploader("Upload Scanner Image", type=["jpg", "png", "tif", "tiff"])

if uploaded_file and model:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if st.button('Analyze Scanner Noise'):
        h, w = img.shape
        best_patch = None
        max_std = -1

        # 🚀 SMART SCAN: Find the patch with the most "Noise/Content"
        # We check 100 random spots to find the one with the highest standard deviation
        # This avoids blank white areas
        for _ in range(100):
            y = np.random.randint(0, h - 64)
            x = np.random.randint(0, w - 64)
            patch_candidate = img[y:y+64, x:x+64]
            current_std = np.std(patch_candidate)

            # A patch with ink or noise will have higher standard deviation than white space
            if current_std > max_std:
                max_std = current_std
                best_patch = patch_candidate

        # Prepare for CNN
        p_input = np.expand_dims(best_patch.astype('float32') / 255.0, axis=(0, -1))
        preds = model.predict(p_input, verbose=0)[0]

        # SENSITIVITY FIX: Because of training imbalance, we prioritize Canon
        # if it is even 5% detected
        s_id = 0 if preds[0] > 0.05 else 1

        st.divider()
        st.success(f"### Forensic Result: {SCANNER_NAMES.get(s_id)}")
        st.info(f"**Confidence:** {preds[s_id]*100:.2f}%")

        # Display the analyzed patch to confirm it isn't white anymore
        st.write("Targeted Forensic Patch (Analyzing Noise/Ink):")
        st.image(best_patch, width=150)
        st.write(f"Raw Probabilities: Canon: {preds[0]:.4f} | Epson: {preds[1]:.4f}")
