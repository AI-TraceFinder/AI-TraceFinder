import cv2
import numpy as np

# Create 10 pairs of images
for i in range(10):
    img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    cv2.imwrite(f'data/input/input_image_{i}.png', img)
    
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(f'data/output/output_image_{i}.png', blurred)

print('Created 10 pairs of sample images successfully')
