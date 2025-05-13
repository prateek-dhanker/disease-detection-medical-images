import tensorflow as tf
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("kidney_stone_model.h5")

# Load and preprocess the image
def preprocess_image(image_path, img_size=(150, 150)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        sys.exit(1)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, img_size)  # Resize to model's expected size
    image = image / 255.0  # Normalize pixel values
    image_expanded = np.expand_dims(image, axis=0)  # Add batch dimension
    
    return image, image_expanded  # Return both original and processed image

# Predict function with graph output
def predict_and_plot(image_path):
    image, processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)[0][0]  # Get single prediction

    # Define labels and confidence scores
    labels = ["Normal", "Kidney Stone"]
    scores = [1 - prediction, prediction]  # Confidence for each class
    
    # Create a plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Show the input image
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Input Image")

    # Show the confidence scores
    axes[1].bar(labels, scores, color=["green", "red"])
    axes[1].set_ylim([0, 1])
    axes[1].set_ylabel("Confidence Score")
    axes[1].set_title("Prediction Confidence")

    # Display the result
    plt.show()

    # Print textual output
    if prediction > 0.5:
        print(f"Prediction: Kidney Stone Detected (Confidence: {prediction:.2f})")
    else:
        print(f"Prediction: Normal (Confidence: {1 - prediction:.2f})")

# Run prediction and plot results
image_path = "yes3.png"  # Change this to your test image path
predict_and_plot(image_path)
