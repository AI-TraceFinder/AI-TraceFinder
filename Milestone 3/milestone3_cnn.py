import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 128
EPOCHS = 10
BATCH_SIZE = 32

LABELS_CSV = "dataset_labels.csv"

print("Loading dataset...")
df = pd.read_csv(LABELS_CSV)
print("Total samples:", len(df))

print("\nAvailable columns:")
print(df.columns)

def load_image(img_path):
    if not os.path.exists(img_path):
        return None
    img = load_img(
        img_path,
        color_mode="grayscale",
        target_size=(IMG_SIZE, IMG_SIZE)
    )
    img = img_to_array(img) / 255.0
    return img

X = []
y = []

for _, row in df.iterrows():
    img_path = row["image_path"]      
    label = row["scanner_model"]      

    img = load_image(img_path)
    if img is not None:
        X.append(img)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Final image array shape:", X.shape)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

print("Classes:", le.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(le.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("\nTraining CNN...")
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

print("\nEvaluating model...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred_classes,
    target_names=le.classes_
))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred_classes)
print(cm)

plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()
