import streamlit as st
import cv2
import pandas as pd
import os
from datetime import datetime
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_image
from utils.predict import predict_scanner
import numpy as np  

st.set_page_config(page_title="Scanner Identification", layout="centered")

st.title("📄 Scanner Identification System")
st.write("Upload a scanned document image to identify its scanner source.")

model = load_model(r"d:\Ai_Tracefinder_Pratik\models\cnn_scanner.h5")  

uploaded_file = st.file_uploader("Upload scanned image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    img = cv2.imdecode(
        np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR
    )

    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    img_pre = preprocess_image(img)
    
    if img_pre.ndim == 4 and img_pre.shape[-1] == 1:
        img_pre = np.repeat(img_pre, 3, axis=-1)

    label, confidence = predict_scanner(model, img_pre)

    st.success(f"🖨️ **Predicted Scanner:** {label}")
    st.info(f"🔢 **Confidence Score:** {confidence:.2f}")

   
    log_dir = r"d:\Ai_Tracefinder_Pratik\logs"
    log_path = os.path.join(log_dir, "features.csv")
    os.makedirs(log_dir, exist_ok=True)

    log_entry = pd.DataFrame([{
        "timestamp": datetime.now(),
        "filename": uploaded_file.name,
        "prediction": label,
        "confidence": confidence
    }])

    if os.path.exists(log_path):
        log_entry.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_entry.to_csv(log_path, index=False)

    
    with open(log_path, "rb") as f:
        csv_bytes = f.read()

    st.download_button(
        "⬇️ Download Prediction Log",
        data=csv_bytes,
        file_name="scanner_predictions.csv",
        mime="text/csv"
    )
