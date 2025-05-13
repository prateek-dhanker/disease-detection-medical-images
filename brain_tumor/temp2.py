import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = tf.keras.models.load_model("brain_tumor_classifier_simple.h5")

# Define the image size
img_size = (128, 128)

# Define class labels
class_labels = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary Tumor"}

# Function to preprocess and predict on a single image
def predict_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    
    # Check if the image was successfully loaded
    if img is None:
        print(f"Error: Could not read the image at {image_path}")
        return None

    # Resize and preprocess
    img_resized = cv2.resize(img, img_size)
    img_norm = img_resized / 255.0  
    img_array = np.expand_dims(img_norm, axis=0)  

    # Make prediction
    predictions = model.predict(img_array)[0]
    predicted_class = np.argmax(predictions)

    # Display the input image
    plt.figure(figsize=(10, 4))
    
    # Plot the input image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Input MRI Image")
    plt.axis("off")
    
    # Plot probabilities of each class
    plt.subplot(1, 2, 2)
    sns.barplot(x=list(class_labels.values()), y=predictions, palette="viridis")
    plt.xlabel("Tumor Type")
    plt.ylabel("Probability")
    plt.title("Probability of Each Tumor Type")
    plt.ylim(0, 1)

    plt.show()

    return predicted_class

# Test the function on a sample image
test_image_path = "test2p.png"  # Change this to your actual image path
predicted_class = predict_image(test_image_path)
