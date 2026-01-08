import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import os

DATASET_DIR = "dataset"
MODEL_PATH = "models/prnu_cnn_final.keras"
IMG_SIZE = 224
BATCH_SIZE = 1

model = tf.keras.models.load_model(MODEL_PATH)

test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    os.path.join(DATASET_DIR, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

preds = model.predict(test_data)
y_pred = np.argmax(preds, axis=1)
y_true = test_data.classes

print("\n🎯 Test Accuracy:", accuracy_score(y_true, y_pred) * 100, "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report:\n",
      classification_report(y_true, y_pred, target_names=list(test_data.class_indices.keys())))



