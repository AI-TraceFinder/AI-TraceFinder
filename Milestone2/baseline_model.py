import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split



BASE_DIR = Path(__file__).resolve().parent
# Original paths supplied by user (do not change these unless you want to override)
ORIG_INPUT_PATH = Path(r"C:\Users\HP\Desktop\sample_input\input_image.png")
ORIG_OUTPUT_PATH = Path(r"C:\Users\HP\Desktop\sample_output\output_image.png")

if ORIG_INPUT_PATH.exists() and ORIG_OUTPUT_PATH.exists():
    INPUT_IMAGE_PATH = ORIG_INPUT_PATH
    OUTPUT_IMAGE_PATH = ORIG_OUTPUT_PATH
else:
    INPUT_IMAGE_PATH = BASE_DIR / "data" / "input" / "input_image.png"
    OUTPUT_IMAGE_PATH = BASE_DIR / "data" / "output" / "output_image.png"


# IMAGE LOADING

def load_gray_image(path):
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found: {path}")
    return img

input_img = load_gray_image(INPUT_IMAGE_PATH)
output_img = load_gray_image(OUTPUT_IMAGE_PATH)


# NOISE EXTRACTION

def extract_noise(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    noise = cv2.absdiff(image, blur)
    return noise

input_noise = extract_noise(input_img)
output_noise = extract_noise(output_img)

# FFT FEATURE EXTRACTION

def fft_features(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    return np.mean(magnitude), np.std(magnitude)


# SIMPLE TEXTURE FEATURE

def texture_feature(image):
    return np.std(image)


# FEATURE VECTOR CREATION

def create_features(image):
    fft_mean, fft_std = fft_features(image)
    texture = texture_feature(image)
    return [fft_mean, fft_std, texture]

X = []
y = []

# Label: 0 = input image, 1 = output image
for _ in range(50):
    X.append(create_features(input_noise))
    y.append(0)

for _ in range(50):
    X.append(create_features(output_noise))
    y.append(1)

X = np.array(X)
y = np.array(y)


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


def train_and_report(model, name, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"===== Results: {name} =====")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


# Train and evaluate SVM
svm = SVC(kernel='linear')
train_and_report(svm, 'SVM (linear)', X_train, X_test, y_train, y_test)

# Train and evaluate Logistic Regression
logreg = LogisticRegression(max_iter=1000)
train_and_report(logreg, 'Logistic Regression', X_train, X_test, y_train, y_test)


# VISUALIZATION

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Input Image Noise")
plt.imshow(input_noise, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Output Image Noise")
plt.imshow(output_noise, cmap='gray')
plt.axis('off')

plt.show()
