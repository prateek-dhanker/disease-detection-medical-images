'''
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

# Load all models
xray_model = load_model("xray_model.h5")  # X-ray model
mri_model = load_model("mri_model.h5")    # MRI model
retina_model = load_model("retina_model.h5")  # Retina model
kidney_stone_model = load_model("kidney_stone_model.h5")  # Kidney stone model

# Image size
IMG_SIZE = (128, 128)

# Class labels for each model
xray_labels = {0: "COVID-19", 1: "NORMAL", 2: "PNEUMONIA", 3: "TUBERCULOSIS"}
mri_labels = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary Tumor"}
retina_labels = {0: "Cataract", 1: "Diabetic Retinopathy", 2: "Glaucoma", 3: "Normal"}
kidney_stone_labels = {0: "Normal", 1: "Kidney Stone Detected"}

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
    img_type = request.form["type"]

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
    elif img_type == "retina":
        model = retina_model
        labels = retina_labels
    elif img_type == "kidney_stone":
        model = kidney_stone_model
        labels = kidney_stone_labels
    else:
        return jsonify({"error": "Invalid type"}), 400

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

if __name__ == "__main__":
    app.run(debug=True)    '''


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



def preprocess_image(img_path, img_size=(128, 128)):
    """Preprocess image for general model prediction."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_resized = cv2.resize(img, img_size)
    img_norm = img_resized / 255.0  
    img_array = np.expand_dims(img_norm, axis=0)  
    return img_array

def preprocess_kidney_image(image_path, img_size=(150, 150)):
    """Preprocess image specifically for kidney stone detection."""
    image = cv2.imread(image_path)
    if image is None:
        return None
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
    
    img_array = preprocess_image(file_path)
    if img_array is None:
        return jsonify({"error": "Invalid image file"}), 400

    if img_type == "xray":
        model = xray_model
        labels = {0: "COVID-19", 1: "NORMAL", 2: "PNEUMONIA", 3: "TUBERCULOSIS"}
    elif img_type == "mri":
        model = mri_model
        labels = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary Tumor"}
    elif img_type == "retina":
        model = retina_model
        labels = {0: "Cataract", 1: "Diabetic Retinopathy", 2: "Glaucoma", 3: "Normal"}
    else:
        return jsonify({"error": "Invalid type"}), 400

    predictions = model.predict(img_array)[0]
    predicted_class = np.argmax(predictions)
    confidence = float(np.max(predictions))
    plot_path = generate_plot(predictions, labels)

    os.remove(file_path)
    
    return jsonify({
        "prediction": labels[predicted_class],
        "confidence": confidence,
        "plot_path": f"/{plot_path}"
    })

if __name__ == "__main__":
    app.run(debug=True)
