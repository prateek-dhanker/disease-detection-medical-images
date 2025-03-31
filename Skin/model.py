import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

# Dataset paths
data_dir = "Skin/2"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Image Parameters
img_size = (128, 128)  
batch_size = 16  
epochs = 20  # Increased epochs

# Data Augmentation to handle class imbalance
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load training and testing data
train_data = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)

# Load MobileNetV2 with fine-tuning
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = True  

# Freeze first 100 layers to retain pre-trained features
for layer in base_model.layers[:100]:
    layer.trainable = False

# Build Model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),  # Increased neurons
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(train_data.class_indices), activation='softmax')  # Dynamic class count
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(train_data, validation_data=test_data, epochs=epochs)

# Save Model
model.save("Skin/skin_disease.h5")
print("Model saved successfully.")

# Print class labels mapping
print("Class Labels Mapping:", train_data.class_indices)
