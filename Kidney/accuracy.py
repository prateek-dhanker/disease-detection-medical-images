import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Define test data directory
test_dir = "Kidney/1/test"  # Adjust path if needed

# Image size and batch size
img_size = (150, 150)
batch_size = 32

# Load the trained model
model = tf.keras.models.load_model("kidney_stone_model.h5")

# Data generator for test set (without augmentation)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False  # Don't shuffle to match true labels
)

# Evaluate model on test data
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Predict on test data
predictions = model.predict(test_generator)
predicted_labels = (predictions > 0.5).astype(int)  # Convert probabilities to class labels
true_labels = test_generator.classes  # Get actual labels

# Classification report
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=["Normal", "Kidney Stone"]))

# Confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Kidney Stone"], yticklabels=["Normal", "Kidney Stone"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
