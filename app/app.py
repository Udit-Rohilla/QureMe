# Import necessary libraries
from flask import Flask, request, render_template, redirect
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = '../model/brain_tumor_model.h5'
model = load_model(MODEL_PATH)
IMG_SIZE = 224

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'static/uploads'  # Updated to include static directory
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Route to handle the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle file uploads and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect('/')
    
    file = request.files['file']
    if file.filename == '':
        return redirect('/')
    
    if file:
        # Save the file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        # Preprocess the image
        img = cv2.imread(filepath)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        
        # Predict using the model
        prediction = model.predict(img)
        result = "Tumor" if prediction > 0.5 else "No Tumor"
        
        # Pass the filename to the result template
        return render_template('result.html', prediction=result, filename=file.filename)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
