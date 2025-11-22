from flask import Flask, request, jsonify
import os
import tensorflow as tf
from werkzeug.utils import secure_filename

# Initialize the Flask application
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'nii', 'nii.gz'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained AI model (replace this with the path to your model)
MODEL_PATH = './model/brain_tumor_detection_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Function to check if the file type is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Endpoint to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']

    # Check if the file is selected and allowed
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    if file and allowed_file(file.filename):
        # Secure the filename and save the file to the uploads folder
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the MRI scan using the AI model
        result = predict_brain_tumor(filepath)

        # Return the prediction result
        return jsonify({"message": "File uploaded successfully", "prediction": result}), 200
    else:
        return jsonify({"error": "Invalid file type"}), 400

# Function to predict brain tumor using the AI model
def predict_brain_tumor(filepath):
    # Load and preprocess the image or MRI scan here
    # Note: Actual preprocessing code depends on your model's input requirements.
    img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension

    # Perform prediction using the model
    prediction = model.predict(img_array)

    # Return a mock prediction (you should adjust this based on your model output)
    return {
        "tumor_detected": bool(prediction[0][0]),
        "probability": float(prediction[0][0])
    }

# Run the Flask app
if __name__ == '__main__':
    # Ensure uploads folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    # Start the server
    app.run(debug=True)
