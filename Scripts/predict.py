import tensorflow as tf
import numpy as np
import cv2
import os

MODEL_PATH = '../model/brain_tumor_model.h5'
IMG_SIZE = 224

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess a single image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to make predictions on a new image
def predict_tumor(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)[0][0]  # Get the prediction
    if prediction > 0.5:
        return "No Tumor"
    else:
        return "Tumor"

# Example usage
if __name__ == "__main__":
    img_path = input("Enter the path of the image to predict: ")
    if os.path.exists(img_path):
        result = predict_tumor(img_path)
        print(f"Prediction: {result}")
    else:
        print("Image file not found.")
