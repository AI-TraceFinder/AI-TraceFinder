import streamlit as st
import numpy as np
import cv2
import os
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="TraceFinder – Scanner Identification",
    page_icon="🖨️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= SESSION STATE =================
if "log" not in st.session_state:
    st.session_state.log = []

# ================= HEADER =================
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem; border-radius: 15px; color: white; text-align:center;">
    <h1 style="margin:0;">🖨️ TraceFinder</h1>
    <p style="margin:0.5rem 0 0 0; opacity: 0.9;">CNN Scanner Identification System</p>
</div>
""", unsafe_allow_html=True)

# ================= LOAD CNN MODEL =================
MODEL_PATH = "/content/drive/MyDrive/Tracefinder/models/cnn_model.h5"
cnn_model = load_model(MODEL_PATH)

# Scanner classes (must match training labels)
CLASS_NAMES = ['Canon120-1', 'EpsonV550', 'HPScanjet', 'Other']  # Example, replace with your classes

# ================= IMAGE PREPROCESS =================
TARGET_SIZE = (224, 224)  # Must match your CNN input size

def preprocess_image(img):
    """
    Resize, normalize, expand dims for CNN prediction
    """
    img_resized = cv2.resize(img, TARGET_SIZE)
    if len(img_resized.shape) == 2:  # grayscale to RGB
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    img_norm = img_resized.astype(np.float32) / 255.0
    img_exp = np.expand_dims(img_norm, axis=0)
    return img_exp

# ================= UPLOAD SECTION =================
st.markdown("""
<div style="text-align: center; margin-bottom: 1rem;">
    <h2>Upload a Scanned Document</h2>
    <p>Supported formats: PNG, JPG, JPEG, TIFF</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(" ", type=["png","jpg","jpeg","tif","tiff"], label_visibility="collapsed")

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        st.error("❌ Unable to read image.")
        st.stop()
    
    # ================= PREDICTION =================
    img_input = preprocess_image(img)
    preds = cnn_model.predict(img_input)
    
    class_idx = np.argmax(preds)
    scanner_pred = CLASS_NAMES[class_idx]
    confidence = float(preds[0][class_idx] * 100)
    
    # ================= DISPLAY =================
    st.image(img, caption=f"Uploaded Image - {img.shape[1]} × {img.shape[0]}", use_container_width=True)
    st.markdown(f"### Detected Scanner: **{scanner_pred}**")
    st.markdown(f"### Confidence: **{confidence:.2f}%**")

    # ================= LOG RESULT =================
    log_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filename": uploaded_file.name,
        "Scanner": scanner_pred,
        "Confidence": round(confidence,2)
    }
    st.session_state.log.append(log_entry)

# ================= HISTORY =================
if st.session_state.log:
    st.markdown("<hr>")
    st.markdown("### 📋 Analysis History")
    log_df = pd.DataFrame(st.session_state.log)
    st.dataframe(log_df)
    csv_data = log_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv_data, file_name="tracefinder_results.csv", mime="text/csv")
