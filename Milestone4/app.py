import streamlit as st
import numpy as np
import cv2
import pickle
import pandas as pd
from datetime import datetime
import os

# Load model
model = pickle.load(open("svm_scanner.pkl", "rb"))

# Scanner class names
class_names = ["HP Scanner", "Canon Scanner", "Epson Scanner", "Brother Scanner"]

# Create Outputs folder if not exists
if not os.path.exists("Outputs"):
    os.mkdir("Outputs")

log_file = "Outputs/predictions.csv"

# Create CSV if not exists
if not os.path.exists(log_file):
    df = pd.DataFrame(columns=["Time", "Prediction", "Confidence"])
    df.to_csv(log_file, index=False)

st.title("📄 Scanner Identification System")
st.write("Upload a scanned image to identify scanner brand/model")

uploaded_file = st.file_uploader(
    "Upload scanned image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    # Read image
    image_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flat = gray.flatten().reshape(1, -1)

    # Prediction
    pred_class = model.predict(flat)[0]
    confidence = np.max(model.predict_proba(flat)) * 100

    predicted_scanner = class_names[pred_class]

    st.success(f"✅ Predicted Scanner: **{predicted_scanner}**")
    st.info(f"📊 Confidence Score: **{confidence:.2f}%**")

    # Log result
    new_row = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Prediction": predicted_scanner,
        "Confidence": f"{confidence:.2f}%"
    }

    df = pd.read_csv(log_file)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(log_file, index=False)

    # Download button
    st.download_button(
        label="⬇ Download Prediction Log",
        data=df.to_csv(index=False),
        file_name="scanner_predictions.csv",
        mime="text/csv"
    )
    st.write("📁 Prediction logged successfully!")
