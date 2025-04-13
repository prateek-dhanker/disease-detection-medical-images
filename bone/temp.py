import tensorflow as tf
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("bone_fracture_finetuned.keras")

# Load and preprocess the image
def preprocess_image(image_path, img_size=(224, 224)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        sys.exit(1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, img_size)             # Resize to 224x224
    image = image / 255.0                           # Normalize to [0, 1]
    image_expanded = np.expand_dims(image, axis=0)  # Add batch dimension

    return image, image_expanded

# Predict function with graph output
def predict_and_plot(image_path):
    image, processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)[0][0]  # Get single prediction

    # Correct label assignment based on training
    labels = ["Fractured", "Not Fractured"]
    scores = [1 - prediction, prediction]  # Fractured = 0, Not Fractured = 1

    # Plot input image and prediction confidence
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Show the input image
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Input Image")

    # Show prediction confidence
    axes[1].bar(labels, scores, color=["red", "green"])
    axes[1].set_ylim([0, 1])
    axes[1].set_ylabel("Confidence Score")
    axes[1].set_title("Prediction Confidence")

    plt.tight_layout()
    plt.show()

    # Print textual result
    if prediction < 0.5:
        print(f"Prediction: Fractured Detected (Confidence: {1 - prediction:.2f})")
    else:
        print(f"Prediction: Not Fractured (Confidence: {prediction:.2f})")

# Run prediction
image_path = "1/test/not fractured/1-rotated1-rotated3-rotated1-rotated1.jpg"  # Change this to your test image path
predict_and_plot(image_path)
