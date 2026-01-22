import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import numpy as np
import cv2
from datetime import datetime

IMAGE_SIZE = 224
MODEL_PATH = "best_scanner_model.pth"

CLASS_NAMES = ["Canon120", "EpsonV39", "EpsonV550", "HP", "NonScanned"]


st.markdown("""
<style>
/* 1. Medium-Tone Professional Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #e2e8f0 0%, #fbcfe8 100%);
    background-attachment: fixed;
}

/* 2. Fix for the Rectangle Box Issue */
/* We remove the bright background and use a subtle border instead */
.card {
    background: rgba(255, 255, 255, 0.4);
    border-radius: 15px;
    border: 2px solid rgba(219, 39, 119, 0.2);
    padding: 20px;
    margin: 10px 0px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
}

/* 3. High-Visibility Typography */
h1 {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    color: #831843; /* Deep Berry - Very visible */
    text-align: center;
    font-weight: 800;
    margin-bottom: 0px !important;
}

.subtitle {
    text-align: center;
    color: #475569; /* Slate Grey */
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 30px;
}

/* 4. Results Styling */
.status-ok {
    color: #be185d; /* Strong Pink */
    font-weight: 800;
    font-size: 26px;
    text-align: center;
    background: white;
    border-radius: 10px;
    padding: 10px;
    border: 1px solid #f9a8d4;
}

.status-no {
    color: #1e293b; /* Deep Navy/Dark Slate */
    font-weight: 800;
    font-size: 26px;
    text-align: center;
    background: white;
    border-radius: 10px;
    padding: 10px;
    border: 1px solid #cbd5e1;
}

/* 5. Clean History Table */
.history {
    background: white;
    border-radius: 12px;
    padding: 10px;
    border: 1px solid #f472b6;
}

/* 6. Metric & Progress Styling */
[data-testid="stMetricValue"] {
    color: #9d174d !important;
}

.stProgress > div > div > div > div {
    background-color: #db2777;
}

/* Fix for general text visibility */
p, label, span {
    color: #1e293b !important;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def generate_gradcam(model, image_tensor, class_idx):
    gradients, activations = [], []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, inp, out):
        activations.append(out)

    layer = model.features[-1]
    layer.register_forward_hook(forward_hook)
    layer.register_backward_hook(backward_hook)

    out = model(image_tensor)
    model.zero_grad()
    out[0, class_idx].backward()

    grad = gradients[0].mean(dim=[1,2])
    act = activations[0][0]
    for i in range(len(grad)):
        act[i] *= grad[i]

    heatmap = act.mean(dim=0).detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()
    return heatmap


def predict_image(image):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out, dim=1)
        conf, pred = torch.max(prob, 1)
    return tensor, pred.item(), conf.item()*100


if "history" not in st.session_state:
    st.session_state.history = []


st.title(" TraceFinder – Scanner Detection")
st.markdown("<div class='subtitle'>AI-powered scanned image forensics</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp","webp"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    tensor, pred_idx, confidence = predict_image(image)
    predicted_class = CLASS_NAMES[pred_idx]

    heatmap = generate_gradcam(model, tensor, pred_idx)
    heatmap = cv2.resize(heatmap, image.size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.image(image, caption="Original", width=280)
    col2.image(overlay, caption="Grad-CAM", width=280)

    st.progress(confidence/100)
    st.metric("Confidence", f"{confidence:.2f}%")

    if predicted_class == "NonScanned":
        st.markdown("<div class='status-no'>❌ NOT Scanned</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-ok'>✅ Scanned</div>", unsafe_allow_html=True)
        st.write(f"🖨 Scanner Brand: **{predicted_class}**")

    st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.history.append({
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Status": "Scanned" if predicted_class!="NonScanned" else "Not Scanned",
        "Brand": predicted_class,
        "Confidence": f"{confidence:.2f}%"
    })

if st.session_state.history:
    st.markdown("## 🕘 Scan History")
    st.markdown('<div class="history">', unsafe_allow_html=True)
    st.table(st.session_state.history)
    st.markdown('</div>', unsafe_allow_html=True)

st.caption("CNN-based Scanner Detection using MobileNetV2")