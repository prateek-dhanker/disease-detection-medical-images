import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set paths
base_dir = "bone/1"
test_dir = "bone/1/test"

# Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Preprocess test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # IMPORTANT for correct predictions alignment
)

# Load the trained model
model = tf.keras.models.load_model("bone/bone_fracture_finetuned.keras")

# Evaluate
loss, accuracy, auc = model.evaluate(test_generator, verbose=1)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")
print(f"📊 Test AUC: {auc:.4f}")
print(f"❌ Test Loss: {loss:.4f}")

# Predict on test data
pred_probs = model.predict(test_generator)
y_pred = (pred_probs > 0.5).astype(int).reshape(-1)  # Binary classification threshold
y_true = test_generator.classes

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
labels = list(test_generator.class_indices.keys())  # ['fractured', 'not fractured']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=labels))
