import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("image_features.csv")

X = df.drop(columns=["label"])
y = df["label"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print("Train size:", X_train.shape[0])
print("Validation size:", X_val.shape[0])
print("Test size:", X_test.shape[0])
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="poly",
        degree=2,
        C=50,
        gamma=0.01,
        coef0=1
    ))
])
model.fit(X_train, y_train)
y_val_pred = model.predict(X_val)

val_accuracy = accuracy_score(y_val, y_val_pred)

print("\n✅ Validation Accuracy:", val_accuracy * 100, "%")

print("\n📊 Validation Classification Report:")
print(classification_report(y_val, y_val_pred))

print("\n🧩 Validation Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))
