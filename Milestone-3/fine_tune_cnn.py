import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os

DATASET_DIR = "dataset"
MODEL_PATH = "models/prnu_cnn_base.keras"
FINAL_MODEL_PATH = "models/prnu_cnn_final.keras"

IMG_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 15

model = tf.keras.models.load_model(MODEL_PATH)

# Lower LR for fine tuning
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    os.path.join(DATASET_DIR, "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

model.save(FINAL_MODEL_PATH)
print("✅ Fine-tuning completed")
