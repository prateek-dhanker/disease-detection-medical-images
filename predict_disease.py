import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = tf.keras.models.load_model('retina_model.h5')

# Define class labels (keeps the original order)
class_labels = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

def preprocess_image(image_path, target_size=(128, 128)):
    """Load and preprocess an image for model prediction."""
    img = cv2.imread(image_path)  # Read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, target_size)  # Resize to match model input
    img = img / 255.0  # Normalize pixel values (0 to 1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_disease(image_path):
    """Predict the disease from an X-ray image and visualize results."""
    img = preprocess_image(image_path)
    predictions = model.predict(img)[0]  # Get prediction values

    predicted_class = np.argmax(predictions)  # Get class index
    confidence = predictions[predicted_class] * 100  # Confidence score
    predicted_label = class_labels[predicted_class]  # Class name

    # Load the original image for visualization
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Create a figure with 2 subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the X-ray image
    axes[0].imshow(img)
    axes[0].axis('off')  # Hide axes
    axes[0].set_title(f"Prediction: {predicted_label}\nConfidence: {confidence:.2f}%", 
                      fontsize=14, fontweight='bold', color='red')

    # Plot the bar graph of class probabilities
    ax = sns.barplot(x=class_labels, y=predictions, ax=axes[1], palette="viridis")

    # Annotate each bar with its probability value
    for i, bar in enumerate(ax.patches):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f"{predictions[i] * 100:.2f}%", 
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

    axes[1].set_xlabel("Disease Type")
    axes[1].set_ylabel("Confidence (%)")
    axes[1].set_title("Prediction Probabilities")
    axes[1].set_ylim(0, 1)  # Ensure scale matches probability (0-1)
    axes[1].tick_params(axis='x', rotation=45)

    # Highlight the most confident prediction
    axes[1].patches[predicted_class].set_color('red')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

# Example Usage - Update with your image path
test_image = "sample_images/catcat.jpg"  # Change to your image path
predict_disease(test_image)