import tensorflow as tf
import numpy as np
import cv2
import os

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = r"C:\Users\M.varshith reddy\Downloads\Milestone-3\models\prnu_cnn_final.keras"
TEST_DIR   = r"C:\Users\M.varshith reddy\Downloads\Milestone-3\dataset\test\Canon"
OUTPUT_DIR = "gradcam_results"
IMG_SIZE = 224

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD MODEL
# =====================================================
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

# Force build
_ = model(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3)))

# =====================================================
# FIND LAST CONV LAYER
# =====================================================
last_conv_layer = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer
        break

if last_conv_layer is None:
    raise ValueError("❌ No Conv2D layer found")

print("✅ Using last conv layer:", last_conv_layer.name)

# =====================================================
# LOAD ONE TEST IMAGE
# =====================================================
files = [f for f in os.listdir(TEST_DIR)
         if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if not files:
    raise ValueError("❌ No test images found")

img_path = os.path.join(TEST_DIR, files[0])
print("✅ Using image:", img_path)

img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
img_array = np.expand_dims(img_resized / 255.0, axis=0)

# =====================================================
# GRAD-CAM (ROBUST VERSION)
# =====================================================
with tf.GradientTape() as tape:
    # Forward pass manually
    x = img_array
    for layer in model.layers:
        if layer == last_conv_layer:
            x = layer(x)
            conv_output = x
            tape.watch(conv_output)
        else:
            x = layer(x)

    # Use logits (pre-softmax)
    logits = x
    class_idx = tf.argmax(logits[0])
    loss = logits[:, class_idx]

# Compute gradients
grads = tape.gradient(loss, conv_output)

if grads is None:
    raise RuntimeError("❌ Gradients are None – Grad-CAM cannot proceed")

# Global average pooling
weights = tf.reduce_mean(grads, axis=(0, 1, 2))

# Compute heatmap
heatmap = tf.reduce_sum(tf.multiply(weights, conv_output[0]), axis=-1)
heatmap = tf.maximum(heatmap, 0)
heatmap /= tf.reduce_max(heatmap) + 1e-8
heatmap = heatmap.numpy()

# =====================================================
# VISUALIZATION
# =====================================================
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.4, 0)

# =====================================================
# SAVE RESULTS
# =====================================================
cv2.imwrite(os.path.join(OUTPUT_DIR, "original.png"),
            cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(OUTPUT_DIR, "heatmap.png"), heatmap_color)
cv2.imwrite(os.path.join(OUTPUT_DIR, "overlay.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print("🎯 Grad-CAM completed successfully")
print("📁 Saved in:", OUTPUT_DIR)
