import tensorflow as tf
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Load the trained model
model = tf.keras.models.load_model("brain_tumor_classifier_simple.h5")

# Define the image size and batch size (same as used during training)
img_size = (128, 128)
batch_size = 16

# Define the testing dataset directory
data_dir = "1"
test_dir = os.path.join(data_dir, "Testing")

# Data Preprocessing
datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_data = datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=False
)

# Get the true labels
y_true = test_data.classes  # Actual class labels
class_labels = list(test_data.class_indices.keys())  # Class names

# Predict the probabilities
y_pred_probs = model.predict(test_data)

# Convert probabilities to class labels
y_pred = np.argmax(y_pred_probs, axis=1)

# Compute classification report
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_labels))

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
