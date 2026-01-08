import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import numpy as np
import os

DATASET_DIR = "dataset"
MODEL_PATH = "models/prnu_cnn_final.keras"
IMG_SIZE = 224
BATCH_SIZE = 1

model = tf.keras.models.load_model(MODEL_PATH)

val_gen = ImageDataGenerator(rescale=1./255)

val_data = val_gen.flow_from_directory(
    os.path.join(DATASET_DIR, "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

preds = model.predict(val_data)
y_pred = np.argmax(preds, axis=1)
y_true = val_data.classes

val_acc = accuracy_score(y_true, y_pred)
print(f"\n🎯 VALIDATION ACCURACY: {val_acc * 100:.2f}%")
