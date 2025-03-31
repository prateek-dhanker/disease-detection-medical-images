import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model_path = "Skin/skin_disease.h5"
if not os.path.exists(model_path):
    print("Error: Model file not found!")
    exit()
    
model = tf.keras.models.load_model(model_path)

# Define the image size
img_size = (128, 128)

# Get class labels dynamically from training data
datagen = ImageDataGenerator(rescale=1./255)
train_data = datagen.flow_from_directory("Skin/2/train", target_size=img_size, batch_size=16, class_mode='categorical')

class_labels = {v: k for k, v in train_data.class_indices.items()}  # Reverse mapping

# Function to preprocess and predict on a single image
def predict_image(image_path):
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read the image at {image_path}")
        return None

    # Resize and normalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img_resized = cv2.resize(img, img_size)
    img_norm = img_resized / 255.0  
    img_array = np.expand_dims(img_norm, axis=0)  

    # Make prediction
    predictions = model.predict(img_array)[0]
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_idx]
    confidence = predictions[predicted_class_idx] * 100  # Convert to percentage

    # Display results
    plt.figure(figsize=(14, 7))  # Increased figure size
    
    # Plot the input image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)", fontsize=14, color='red')
    plt.axis("off")
    
    # Plot probabilities of each class
    plt.subplot(1, 2, 2)
    sns.barplot(x=list(class_labels.values()), y=predictions, palette="viridis")
    plt.xlabel("Skin Disease Type")
    plt.ylabel("Probability")
    plt.title("Probability of Each Disease Type")

    # Improved readability of labels
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.ylim(0, 1)

    # Adjust layout to fit all labels
    plt.tight_layout()

    plt.show()

    return predicted_class


# Test the function on a sample image
test_image_path = "Skin/2/test/Basal Cell Carcinoma/ISIC_0070682.jpg"
predicted_class = predict_image(test_image_path)
print("Predicted Class:", predicted_class)
