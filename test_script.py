import numpy as np
import pandas as pd
import os
import cv2
import gc
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model

# Function to crop images and remove unnecessary dark regions
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        if img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0] == 0:
            return img  # Return original image if too dark
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

# Function for image preprocessing
def preprocess_image(image_path, desired_size=256):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (desired_size, desired_size))
    img = cv2.addWeighted(img, 4.5, cv2.GaussianBlur(img, (0, 0), 10), -4, 100)
    return img

# Load test dataset
print("Loading test dataset...")
test_df = pd.read_csv('/content/dataset/test.csv')

# Preprocess test images
N_test = test_df.shape[0]
x_test = np.empty((N_test, 256, 256, 3), dtype=np.uint8)

for i, image_id in tqdm(enumerate(test_df['id_code'])):
    image_path = f'/content/dataset/test_images/{image_id}.png'
    x_test[i] = preprocess_image(image_path)

# Load trained model
model = load_model('/content/dataset/efficientnetb7_model.h5')

# Make predictions
y_pred = model.predict(x_test)
predicted_labels = np.argmax(y_pred, axis=1)

# Save predictions
test_df['diagnosis'] = predicted_labels
test_df.to_csv('/content/dataset/test_predictions.csv', index=False)

print("Test predictions saved successfully!")

# Clear memory
gc.collect()

print("Test dataset preprocessed and predictions saved successfully!")