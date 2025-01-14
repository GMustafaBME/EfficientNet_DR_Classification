# this code is going to take input images of aptos2019 dataset
# and train EfficientNet B7
# Check the location of the files if they are correct
import numpy as np
import pandas as pd
import os
import cv2
import gc
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load dataset
train_df = pd.read_csv('/content/dataset/train.csv') #Provide your train data labelling location

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

# Preprocess and load images
N = train_df.shape[0]
x_data = np.empty((N, 256, 256, 3), dtype=np.uint8)
y_data = train_df['diagnosis'].values

for i, image_id in tqdm(enumerate(train_df['id_code'])):
    image_path = f'/content/dataset/train_images/{image_id}.png'
    x_data[i] = preprocess_image(image_path)

# Convert labels to categorical
num_classes = len(np.unique(y_data))
y_data = to_categorical(y_data, num_classes)

# Split dataset into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42, stratify=y_data)

# Build EfficientNetB7 Model
base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
base_model.trainable = False  # Freeze base model layers

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output_layer = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output_layer)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=32,
    verbose=1
)

# Save trained model
model.save('/content/dataset/efficientnetb7_model.h5')

# Clear memory
gc.collect()

print("Dataset preprocessed, split, model trained, and saved successfully!")
