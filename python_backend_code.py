import os
import uuid
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, render_template, send_file
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Ensure the 'static/' directory exists
if not os.path.exists("static"):
    os.makedirs("static")

# Load both models
xray_model = load_model("xray_model.h5")  # X-ray model
mri_model = load_model("mri_model.h5")    # MRI model

# Image size
IMG_SIZE = (128, 128)

# Class labels for both models
xray_labels = {0: "COVID-19", 1: "NORMAL", 2: "PNEUMONIA", 3: "TUBERCLUSOSIS"}
mri_labels = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary Tumor"}

def preprocess_image(img_path):
    """Preprocess image for model prediction."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    img_resized = cv2.resize(img, IMG_SIZE)
    img_norm = img_resized / 255.0  
    img_array = np.expand_dims(img_norm, axis=0)  

    return img_array

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



@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and return model prediction."""
    if "file" not in request.files or "type" not in request.form:
        return jsonify({"error": "File and type are required"}), 400
    
    file = request.files["file"]
    img_type = request.form["type"]  # "xray" or "mri"

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join("static", filename)
    file.save(file_path)

    img_array = preprocess_image(file_path)
    if img_array is None:
        return jsonify({"error": "Invalid image file"}), 400

    # Select the model and class labels based on the type
    if img_type == "xray":
        model = xray_model
        labels = xray_labels
    elif img_type == "mri":
        model = mri_model
        labels = mri_labels
    else:
        return jsonify({"error": "Invalid type, must be 'xray' or 'mri'"}), 400

    predictions = model.predict(img_array)[0]
    predicted_class = np.argmax(predictions)  
    confidence = float(np.max(predictions))  

    plot_path = generate_plot(predictions, labels)

    os.remove(file_path)

    result = {
        "prediction": labels[predicted_class],
        "confidence": confidence,
        "plot_path": f"/{plot_path}"
    }
    return jsonify(result)

# This code for result image show in box
@app.route("/plot/<filename>")
def get_plot(filename):
    # """Serve the generated probability plot."""
    return send_file(os.path.join("static", filename), mimetype="image/png")



if __name__ == "__main__":
    app.run(debug=True)
