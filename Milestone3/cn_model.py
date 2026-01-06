import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib
# Use non-interactive backend so script can run in headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# PATHS (CHANGE ONLY THIS)

DATASET_PATH = "data"   # Folder containing scannerA, scannerB, etc.
IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 10


# DATA LOADING + AUGMENTATION

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    brightness_range=[0.8, 1.2]
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

num_classes = train_data.num_classes
class_names = list(train_data.class_indices.keys())


# CNN MODEL (WEEK 5)

model = tf.keras.Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# ACCURACY & LOSS PLOTS

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label="Train")
plt.plot(history.history['val_accuracy'], label="Val")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label="Train")
plt.plot(history.history['val_loss'], label="Val")
plt.title("Loss")
plt.legend()

plt.savefig('training_plots.png', bbox_inches='tight')
plt.close()


# CONFUSION MATRIX (WEEK 6)

y_true = val_data.classes
y_pred = np.argmax(model.predict(val_data), axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Scanner Identification")
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.close()

print(classification_report(y_true, y_pred, target_names=class_names))


# GRAD-CAM (EXPLAINABILITY)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # Allow passing None or an incorrect layer name: find last Conv2D if needed
    last_conv = None
    if last_conv_layer_name:
        try:
            last_conv = model.get_layer(last_conv_layer_name)
        except (ValueError, AttributeError):
            last_conv = None

    if last_conv is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer
                last_conv_layer_name = layer.name
                break

    if last_conv is None:
        raise ValueError("No Conv2D layer found in the model for Grad-CAM.")

    # Build a functional model mapping a new Input to the last conv output and final predictions
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x_tensor = inp
    conv_outputs_tensor = None
    for layer in model.layers:
        x_tensor = layer(x_tensor)
        if layer.name == last_conv_layer_name:
            conv_outputs_tensor = x_tensor
    preds_tensor = x_tensor
    grad_model = Model(inp, [conv_outputs_tensor, preds_tensor])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    # Weighted combination of filters
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # Ensure non-negative and safe normalization
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    max_val = max_val.numpy() if hasattr(max_val, "numpy") else float(max_val)
    if max_val == 0 or np.isnan(max_val):
        return np.zeros((heatmap.shape[0], heatmap.shape[1]))

    heatmap = heatmap / max_val
    return heatmap.numpy()

# Take one validation image
img_path = val_data.filepaths[0]
img = cv2.imread(img_path)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img_array = np.expand_dims(img/255.0, axis=0)

heatmap = make_gradcam_heatmap(img_array, model, "conv2d_1")

# Overlay heatmap
heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

plt.figure(figsize=(6,6))
plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
plt.title("Grad-CAM - Scanner Pattern Focus")
plt.axis("off")
plt.savefig('gradcam.png', bbox_inches='tight')
plt.close()
