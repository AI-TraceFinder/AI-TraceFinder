import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------------------------
# Load Extracted Features
# -------------------------------------------
DATA_FILE = "/content/drive/MyDrive/a/Output/features.csv"
df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)

# -------------------------------------------
# Prepare Data
# -------------------------------------------
X = df.drop(columns=["image_path", "label"])
y = df["label"]

# Encode string labels
le = LabelEncoder()
y = le.fit_transform(y)

# Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------
# Feature Scaling
# -------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------------------
# Train Model
# -------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------------------------------------------
# Evaluation
# -------------------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

import joblib

joblib.dump(model, "scanner_classifier.pkl")
joblib.dump(scaler, "feature_scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

def predict_image(image_path):
    features = build_feature_vector(image_path)
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)

    pred = model.predict(features)[0]
    return le.inverse_transform([pred])[0]
