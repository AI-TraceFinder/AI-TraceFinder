import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("=== Milestone 2: Baseline Modeling ===")

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_CSV = os.path.join(BASE_DIR, "milestone2", "features.csv")

# --------------------------------------------------
# Load features
# --------------------------------------------------
df = pd.read_csv(FEATURES_CSV)
print(f"[INFO] Loaded feature matrix with shape: {df.shape}")
print(f"[INFO] Columns: {list(df.columns)}")

# --------------------------------------------------
# Target label (WHAT we classify)
# --------------------------------------------------
# Choose ONE of these (scanner is best)
TARGET_COL = "scanner"  # or "source_type"

# --------------------------------------------------
# Feature selection
# --------------------------------------------------
# Keep only numeric columns for ML
X = df.select_dtypes(include=[np.number])

# Labels
y = df[TARGET_COL]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

print(f"[INFO] Classes: {list(le.classes_)}")

# --------------------------------------------------
# Train / Test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# Logistic Regression
# --------------------------------------------------
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)

lr_acc = accuracy_score(y_test, y_pred_lr)
print(f"\n[RESULT] Logistic Regression Accuracy: {lr_acc:.4f}")

# --------------------------------------------------
# Random Forest
# --------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rf_acc = accuracy_score(y_test, y_pred_rf)
print(f"[RESULT] Random Forest Accuracy: {rf_acc:.4f}")

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
print("\n[CONFUSION MATRIX - RF]")
print(confusion_matrix(y_test, y_pred_rf))

print("\n[CLASSIFICATION REPORT - RF]")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# --------------------------------------------------
# Save results
# --------------------------------------------------
RESULTS_PATH = os.path.join(BASE_DIR, "milestone2", "results.txt")
with open(RESULTS_PATH, "w") as f:
    f.write(f"Logistic Regression Accuracy: {lr_acc:.4f}\n")
    f.write(f"Random Forest Accuracy: {rf_acc:.4f}\n")

print(f"\n[OK] Results saved to: {RESULTS_PATH}")

