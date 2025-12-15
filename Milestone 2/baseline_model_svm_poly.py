import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df = pd.read_csv("image_features.csv")

X = df.drop(columns=["label"])
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="poly"))
])
param_grid = {
    "svm__degree": [2, 3],
    "svm__C": [0.1, 1, 10, 50, 100],
    "svm__gamma": ["scale", 0.01, 0.001],
    "svm__coef0": [0, 1]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

print("⏳ Training Polynomial SVM...")
grid.fit(X_train, y_train)

print("✅ Best Parameters:", grid.best_params_)
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\n🎯 Test Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
