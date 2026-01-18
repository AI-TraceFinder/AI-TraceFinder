import pickle
import numpy as np
from sklearn.svm import SVC

# Create dummy training data
# 128x128 grayscale images = 16384 features
n_samples_per_class = 20
n_features = 128 * 128  # 16384

# Generate random training data for 4 scanner classes
X_train = []
y_train = []

for class_id in range(4):
    # Generate random samples for each class
    samples = np.random.randint(0, 256, (n_samples_per_class, n_features))
    X_train.append(samples)
    y_train.extend([class_id] * n_samples_per_class)

X_train = np.vstack(X_train)
y_train = np.array(y_train)

# Train SVM model
print("Training SVM model...")
model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("svm_scanner.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as 'svm_scanner.pkl'")
print(f"Model trained on {len(X_train)} samples with {n_features} features")
print(f"Number of classes: 4 (Scanner A, B, C, D)")
