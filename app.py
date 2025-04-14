import os
import uuid
import numpy as np
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, render_template, send_file
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Ensure the 'static/' directory exists
if not os.path.exists("static"):
    os.makedirs("static")

# Load all models
xray_model = load_model("xray_model.h5")  # X-ray model
mri_model = load_model("mri_model.h5")    # MRI model
retina_model = load_model("retina_model.h5")  # Retina model
kidney_stone_model = load_model("kidney_stone_model.h5")  # Kidney stone model
bone_fracture_model = load_model("bone_model.keras")



def preprocess_image(img_path, img_size=(128, 128)):
    """Preprocess image for general model prediction."""
    img = cv2.imread(img_path)
    if img is None:
        return jsonify({"error": "Invalid image file"}), 400
    
    img_resized = cv2.resize(img, img_size)
    img_norm = img_resized / 255.0  
    img_array = np.expand_dims(img_norm, axis=0)  
    return img_array

def preprocess_kidney_image(image_path, img_size=(150, 150)):
    """Preprocess image specifically for kidney stone detection."""
    image = cv2.imread(image_path)
    if image is None:
        return jsonify({"error": "Invalid image file"}), 400
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, img_size)
    image = image / 255.0
    image_expanded = np.expand_dims(image, axis=0)
    return image_expanded

def generate_plot(predictions, labels):
    """Generate and save a probability plot."""
    plot_filename = f"probability_plot_{uuid.uuid4().hex}.png"
    plot_path = os.path.join("static", plot_filename)
    
    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(labels.values()), y=predictions, palette="viridis")
    plt.xlabel("Disease Type")
    plt.ylabel("Probability")
    plt.title("Probability of Each Class")
    plt.ylim(0, 1)
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def predict_kidney_stone(image_path):
    """Perform kidney stone detection using specific logic."""

    # Process the image
    processed_image = preprocess_kidney_image(image_path)
    prediction = kidney_stone_model.predict(processed_image)[0][0]
    
    labels = ["Normal", "Kidney Stone"]
    scores = [1 - prediction, prediction]
    
    # Generate unique filename for the result image
    plot_filename = f"kidney_prediction_{uuid.uuid4().hex}.png"
    plot_path = os.path.join("static", plot_filename)

    # Create figure with input image and prediction confidence
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.imread(image_path)[:, :, ::-1])  # Convert BGR to RGB
    axes[0].axis("off")
    axes[0].set_title("Input Image")

    axes[1].bar(labels, scores, color=["green", "red"])
    axes[1].set_ylim([0, 1])
    axes[1].set_ylabel("Confidence Score")
    axes[1].set_title("Prediction Confidence")

    # Save the generated plot
    plt.savefig(plot_path)
    plt.close()

    return {
        "prediction": "Kidney Stone Detected" if prediction > 0.5 else "Normal",
        "confidence": float(prediction) if prediction > 0.5 else float(1 - prediction),
        "plot_path": f"/static/{plot_filename}"  # Ensure unique path
    }

def predict_disease(image_path,model_name,labels):
    """Perform kidney stone detection using specific logic."""

    # Process the image
    processed_image = preprocess_image(image_path)
    predictions = model_name.predict(processed_image)[0]

    # Get predicted class and confidence
    predicted_class = np.argmax(predictions)
    confidence = float(predictions[predicted_class])
    
    # Generate unique filename for the result image
    plot_filename = f"xray_prediction_{uuid.uuid4().hex}.png"
    plot_path = os.path.join("static", plot_filename)

    # Create figure with input image and prediction confidence
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.imread(image_path)[:, :, ::-1])  # Convert BGR to RGB
    axes[0].axis("off")
    axes[0].set_title("Input Image")

    axes[1].bar(labels, predictions, color=["green", "red", "blue", "orange"])
    axes[1].set_ylim([0, 1])
    axes[1].set_ylabel("Confidence Score")
    axes[1].set_title("Prediction Confidence")

    # Save the generated plot
    plt.savefig(plot_path)
    plt.close()

    return {
        "prediction": labels[predicted_class],
        "confidence": confidence,
        "plot_path": f"/static/{plot_filename}"  # Ensure unique path
    }


def preprocess_bone_image(image_path, img_size=(224, 224)):
    """Preprocess image specifically for bone fracture detection."""
    image = cv2.imread(image_path)
    if image is None:
        return jsonify({"error": "Invalid image file"}), 400
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, img_size)
    image = image / 255.0
    image_expanded = np.expand_dims(image, axis=0)
    return image_expanded

def predict_bone_fracture_s(image_path):
    """Perform bone fracture detection using specific logic."""
    
    # Process the image
    processed_image = preprocess_bone_image(image_path)
    prediction = bone_fracture_model.predict(processed_image)[0][0]

    labels = ["Fractured", "Not Fractured"]
    scores = [1 - prediction, prediction]

    # Generate unique filename for the result image
    plot_filename = f"bone_prediction_{uuid.uuid4().hex}.png"
    plot_path = os.path.join("static", plot_filename)

    # Create figure with input image and prediction confidence
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.imread(image_path)[:, :, ::-1])  # Convert BGR to RGB
    axes[0].axis("off")
    axes[0].set_title("Input Image")

    axes[1].bar(labels, scores, color=["red", "green"])
    axes[1].set_ylim([0, 1])
    axes[1].set_ylabel("Confidence Score")
    axes[1].set_title("Prediction Confidence")

    # Save the generated plot
    plt.savefig(plot_path)
    plt.close()

    return {
        "prediction": "Fractured" if prediction < 0.5 else "Not Fractured",
        "confidence": float(1 - prediction if prediction < 0.5 else prediction),
        "plot_path": f"/static/{plot_filename}"  # Ensure unique path
    }


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/diagnosis")
def diagnosis():
    return render_template("diagnosis.html")

@app.route("/blog")
def blog():
    return render_template("blog.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact_us")
def contact_us():
    return render_template("contact_us.html")

@app.route("/feedback")
def feedback():
    return render_template("feedback.html")

@app.route("/submit_feedback")
def submit_feedback():
    return render_template("feedback_reaction.html")

@app.route("/result")
def result_page():
    return render_template("result.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and return model prediction."""
    if "file" not in request.files or "type" not in request.form:
        return jsonify({"error": "File and type are required"}), 400
    
    file = request.files["file"]
    img_type = request.form["type"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join("static", filename)
    file.save(file_path)

    if img_type == "kidney_stone":
        result = predict_kidney_stone(file_path)
        os.remove(file_path)
        return jsonify(result)
    
    elif img_type == "xray":
        labels = ["COVID-19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
        result = predict_disease(file_path,xray_model,labels)
        os.remove(file_path)
        return jsonify(result)
    
    elif img_type == "retina":
        labels = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]
        result = predict_disease(file_path,retina_model,labels)
        os.remove(file_path)
        return jsonify(result)
    
    elif img_type == "mri":
        labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
        result = predict_disease(file_path,mri_model,labels)
        os.remove(file_path)
        return jsonify(result)
    
    elif img_type == "bone":
        result = predict_bone_fracture_s(file_path)
        os.remove(file_path)
        return jsonify(result)

    
    else:
        return jsonify({"error": "Invalid type"}), 400

if __name__ == "__main__":
    app.run(debug=True)
