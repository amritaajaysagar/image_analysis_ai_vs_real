import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
import cv2
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Path to your saved model
MODEL_PATH = 'saved_model'  # Ensure this is the path to your 'saved_model' directory

# Load the SavedModel
model = tf.saved_model.load(MODEL_PATH)

# Define upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = tf.image.resize(img, (256, 256))  # Resize to match model input shape
    img = img.numpy().astype("float32") / 255.0  # Normalize to [0,1]
    return np.expand_dims(img, axis=0)  # Expand dims to match model input shape

# Define homepage route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get uploaded file
        if "file" not in request.files:
            return render_template("index.html", prediction="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", prediction="No file selected")

        # Save file to upload folder
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Preprocess and predict
        processed_img = preprocess_image(file_path)

        # Assuming the model's signature is "serving_default" (default behavior for most models)
        # Perform inference using the model
        infer = model.signatures["serving_default"]
        input_tensor = tf.convert_to_tensor(processed_img)  # Convert input image to tensor
        output = infer(input_tensor)

        # Assuming the output is a single value (like classification score)
        yhat = output["output_0"].numpy()[0]  # Adjust the key based on your model's output

        # Determine class label
        prediction = "Real" if yhat > 0.5 else "AI-Generated"
        
        return render_template("index.html", prediction=prediction, image_url=file_path)

    return render_template("index.html", prediction=None)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
