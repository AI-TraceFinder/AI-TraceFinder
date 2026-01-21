import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
import datetime
import tensorflow as tf
from PIL import Image

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="TraceFinder - Scanner Identification",
    layout="centered"
)

MODEL_PATH = "models/prnu_cnn_final.keras"
LOG_FILE = "logs/predictions.csv"
IMG_SIZE = 224   # change ONLY if your training used different size

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r"C:\Users\M.varshith reddy\Downloads\Nageswari-Mettukuru\Milestone-4\models\prnu_cnn_final.keras")


model = load_model()

# -------------------- LABEL MAP --------------------
label_map = {
    0: "HP Scanner",
    1: "Canon Scanner",
    2: "Epson Scanner",
    3: "Brother Scanner"
}

# -------------------- IMAGE PREPROCESS --------------------
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -------------------- PREDICTION --------------------
def predict_scanner(image):
    processed = preprocess_image(image)
    preds = model.predict(processed)
    class_id = np.argmax(preds)
    confidence = float(np.max(preds) * 100)
    return label_map[class_id], round(confidence, 2)

# -------------------- UI --------------------
st.title("🕵️ TraceFinder – Scanner Identification")
st.write("Upload a scanned image to identify the scanner brand/model.")

uploaded_file = st.file_uploader(
    "Upload Scanned Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict Scanner"):
        img_array = np.array(image)
        scanner, confidence = predict_scanner(img_array)

        st.success(f"**Predicted Scanner:** {scanner}")
        st.info(f"**Confidence Score:** {confidence}%")

        # -------------------- LOGGING --------------------
        os.makedirs("logs", exist_ok=True)

        log_data = {
            "Timestamp": datetime.datetime.now(),
            "Image_Name": uploaded_file.name,
            "Predicted_Scanner": scanner,
            "Confidence (%)": confidence
        }

        df = pd.DataFrame([log_data])

        if os.path.exists(LOG_FILE):
            df.to_csv(LOG_FILE, mode="a", header=False, index=False)
        else:
            df.to_csv(LOG_FILE, index=False)

        st.success("Prediction logged successfully!")

# -------------------- DOWNLOAD LOG --------------------
if os.path.exists(LOG_FILE):
    st.download_button(
        label="⬇ Download Prediction Log",
        data=pd.read_csv(LOG_FILE).to_csv(index=False),
        file_name="scanner_predictions.csv",
        mime="text/csv"
    )
