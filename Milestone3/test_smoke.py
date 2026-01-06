import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

IMG_SIZE = 128

# Small model matching shapes in cn_model.py
model = tf.keras.Sequential([
    Conv2D(8, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(2, activation='softmax')
])

# Run one forward pass to build weights
x = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
_ = model.predict(x)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
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

    # Build a functional model that maps the input to the last conv output and final predictions
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

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    max_val = max_val.numpy() if hasattr(max_val, "numpy") else float(max_val)
    if max_val == 0 or np.isnan(max_val):
        print('Heatmap is empty or invalid')
        print('shape:', heatmap.shape)
        raise SystemExit(1)

    heatmap = heatmap / max_val
    return heatmap.numpy()

# Run heatmap
heatmap = make_gradcam_heatmap(x, model, None)
print('Heatmap OK — shape:', heatmap.shape, 'min/max:', heatmap.min(), heatmap.max())
