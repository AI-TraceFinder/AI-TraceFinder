import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# IMAGE PATHS 
input_image_path = "C:\\Users\\HP\\Desktop\\sample_input\\input.tif"
output_image_path = "C:\\Users\\HP\\Desktop\\sample_output\\ouput.png"



# LOAD IMAGES

input_img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
output_img = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)

if input_img is None or output_img is None:
    raise FileNotFoundError("Check image paths or filenames.")

input_img = cv2.resize(input_img, (256, 256))
output_img = cv2.resize(output_img, (256, 256))



# NOISE EXTRACTION FUNCTION

def extract_noise(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    noise = cv2.absdiff(image, blur)
    return noise


input_noise = extract_noise(input_img)
output_noise = extract_noise(output_img)



# DISPLAY NOISE MAPS

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(input_noise, cmap="gray")
plt.title("Input Image Noise")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output_noise, cmap="gray")
plt.title("Output Image Noise")
plt.axis("off")

plt.show()



# FFT FEATURE EXTRACTION

def fft_features(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    return magnitude.mean(), magnitude.std()


input_fft = fft_features(input_img)
output_fft = fft_features(output_img)


# LBP FEATURE EXTRACTION

def lbp_features(image):
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


input_lbp = lbp_features(input_img)
output_lbp = lbp_features(output_img)



X = []
y = []

# Input image = class 0, Output image = class 1
X.append(list(input_fft) + list(input_lbp))
y.append(0)

X.append(list(output_fft) + list(output_lbp))
y.append(1)

X = np.array(X)
y = np.array(y)

print(f"\nDataset created with {len(X)} samples")
print(f"Feature vector size: {X.shape[1]}")



print("\nFeature Comparison:")
print(f"Input Image - FFT mean: {input_fft[0]:.2f}, FFT std: {input_fft[1]:.2f}")
print(f"Output Image - FFT mean: {output_fft[0]:.2f}, FFT std: {output_fft[1]:.2f}")
print(f"LBP histogram difference (L1 norm): {np.sum(np.abs(input_lbp - output_lbp)):.4f}")


print("\n" + "="*50)
print("Note: With only 2 images, proper ML training is not feasible.")
print("This demonstrates feature extraction only.")
print("="*50)


print("\nMilestone 2 execution completed successfully.")

