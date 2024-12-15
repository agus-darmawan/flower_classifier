import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import RandomFlip

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = r'D:\Flowerr\Flowerrecognew.h5'  # Update with your correct model path

# Create necessary folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define custom objects for loading the model
custom_objects = {
    'RandomFlip': RandomFlip
}

# Load model
model = load_model(MODEL_PATH, custom_objects=custom_objects)

# Flower names (output classes)
flower_names = ['bougainville', 'daisy', 'dandelion', 'lily', 'rose', 'sunflower', 'tulip']

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def classify_image(image_path):
    """Classify the uploaded image using the pre-trained model."""
    try:
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = np.expand_dims(input_image_array, axis=0)

        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])
        predicted_class = np.argmax(result)
        predicted_score = np.max(result) * 100

        outcome = f"The Image belongs to {flower_names[predicted_class]} with a score of {predicted_score:.2f}%"
        return outcome
    except Exception as e:
        raise Exception(f"Error in image classification: {str(e)}")

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handle image uploads and predict the flower class."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        # Classify the uploaded image
        prediction = classify_image(file_path)

        # Path to display the uploaded image on the webpage
        uploaded_image_url = f'/uploads/{filename}'

        return render_template('index.html',
                               uploaded_image=uploaded_image_url,
                               prediction=prediction)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve the uploaded file from the server."""
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5010)
