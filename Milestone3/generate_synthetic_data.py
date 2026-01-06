import os
import numpy as np
import cv2

IMG_SIZE = 224
NUM_PER_CLASS = 40

classes = ['class0', 'class1']

for cls in classes:
    dirpath = os.path.join('data', cls)
    os.makedirs(dirpath, exist_ok=True)
    for i in range(NUM_PER_CLASS):
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        if cls == 'class0':
            cv2.rectangle(img, (50, 50), (IMG_SIZE-50, IMG_SIZE-50), (0, 0, 255), -1)
        else:
            cv2.circle(img, (IMG_SIZE//2, IMG_SIZE//2), 60, (0, 255, 0), -1)
        path = os.path.join(dirpath, f"{cls}_{i}.jpg")
        cv2.imwrite(path, img)

print('Synthetic dataset created with classes:', classes)
