import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
FEATURES_CSV = "features.csv"
print("\nLoading features.csv...")
df = pd.read_csv(FEATURES_CSV)
print("Total feature rows:", len(df))
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print("Rows after cleaning:", len(df))
X = df.drop("scanner", axis=1)
y = df["scanner"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
# MODEL 1: DECISION TREE
print("\nTraining Decision Tree...")
dt = DecisionTreeClassifier(
    max_depth=20,
    class_weight="balanced",
    random_state=42
)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
# MODEL 2: LOGISTIC REGRESSION
print("\nTraining Logistic Regression...")
lr = LogisticRegression(
    max_iter=5000,
    solver="lbfgs",
    class_weight="balanced"
)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n================ {name} RESULTS ================\n")
    print(f"Accuracy  : {acc:.4f}")
    print(f"RMSE      : {rmse:.4f}")
    print(f"R² Score  : {r2:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    return acc
dt_acc = evaluate_model("Decision Tree", y_test, dt_pred)
lr_acc = evaluate_model("Logistic Regression", y_test, lr_pred)
best_model = "Decision Tree" if dt_acc > lr_acc else "Logistic Regression"
best_acc = max(dt_acc, lr_acc)
print("\nBEST MODEL:", best_model)
print(f"Best Accuracy: {best_acc:.4f}")

