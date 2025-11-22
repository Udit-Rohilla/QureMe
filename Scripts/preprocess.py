import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

IMG_SIZE = 224  # Size to resize images
data_dir = '../data/'  # Dataset directory

# Load and preprocess images
def load_images(data_dir, img_size):
    images = []
    labels = []
    for label, class_dir in enumerate(['NoTumor', 'Tumor']):
        class_dir_path = os.path.join(data_dir, class_dir)
        for img_file in os.listdir(class_dir_path):  
            img_path = os.path.join(class_dir_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype('float32') / 255.0     # Convert to float32 and normalize between 0 and 1
            images.append(img)
            labels.append(0 if class_dir == 'NoTumor' else 1)
    return np.array(images), np.array(labels)

# Split dataset into training and validation sets
def preprocess_data():
    images, labels = load_images(data_dir, IMG_SIZE)  
    images = images / 255.0  # Normalize pixel values
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

if __name__ == '__main__':
    X_train, X_val, y_train, y_val = preprocess_data()
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
