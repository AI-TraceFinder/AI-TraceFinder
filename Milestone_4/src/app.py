%%writefile app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

# 1. Page Configuration (Native)
st.set_page_config(page_title="Forensic TraceFinder", layout="wide")

# 2. Cached Model Loading (Prevents site from freezing)
@st.cache_resource
def load_analysis_model():
    model_path = "models/best_ai_tracefinder.keras"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_analysis_model()
SCANNERS = ["Canon9000-1", "EpsonV550", "HP"]

# 3. Sidebar Information
with st.sidebar:
    st.title("🛡️ System Info")
    st.success("AI Engine: Active")
    st.markdown("---")
    st.write("**Milestone 4 Deployment**")
    st.write("Targeting PRNU sensor noise patterns.")

# 4. Main Interface
st.title("🕵️‍♂️ Forensic TraceFinder AI")
st.caption("Advanced Identification Protocol for Document Scanners")

# Use columns for a balanced layout
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.subheader("Step 1: Upload Evidence")
    uploaded_file = st.file_uploader("Choose a scan patch (PNG/JPG)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        # Display preview immediately
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        st.image(img, caption="Uploaded Original", use_container_width=True)

with col_result:
    st.subheader("Step 2: AI Analysis")
    
    if uploaded_file is None:
        st.info("Please upload a scan file to begin identification.")
    elif model is None:
        st.error("AI Model not found. Check /models directory.")
    else:
        # Analysis Logic
        with st.spinner("Extracting noise signatures..."):
            # Center Crop 64x64
            h, w = img.shape
            patch = img[h//2-32 : h//2+32, w//2-32 : w//2+32]
            
            # Prepare for Model
            img_input = patch.astype('float32') / 255.0
            img_input = np.expand_dims(img_input, axis=(0, -1))
            
            # Prediction
            preds = model.predict(img_input)
            confidence = np.max(preds[0])
            idx = np.argmax(preds[0])
            
            # Results UI
            st.toast("Analysis Complete!")
            st.metric(label="Predicted Brand", value=SCANNERS[idx], delta=f"{confidence*100:.2f}% Match")
            
            st.progress(float(confidence))
            
            with st.expander("View Forensic Patch"):
                st.image(patch, caption="Isolated 64x64 Sensor Noise", width=150)
            
            if confidence > 0.80:
                st.success("Verified Match: High confidence in sensor signature.")
            else:
                st.warning("Low Confidence: Signature may be degraded or device is unknown.")