import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
import os

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = '../model/brain_tumor_model.h5'
DATA_DIR = '../data'  # Ensure this points to your data directory

# Data augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale = 1./255,  # Normalize pixel values between 0 and 1
    rotation_range = 10,
    zoom_range = 0.1,
    horizontal_flip = True,
    validation_split = 0.2  # Split for validation
)

# Load training data in batches
train_gen = train_datagen.flow_from_directory(
    DATA_DIR, 
    target_size = (IMG_SIZE, IMG_SIZE), 
    batch_size = BATCH_SIZE, 
    class_mode = 'binary',
    subset = 'training'
)

# Load validation data in batches (no augmentation)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_gen = val_datagen.flow_from_directory(
    DATA_DIR, 
    target_size = (IMG_SIZE, IMG_SIZE), 
    batch_size = BATCH_SIZE, 
    class_mode = 'binary',
    subset = 'validation'
)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation = 'relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation = 'relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(1, activation = 'sigmoid')  # Binary classification (tumor/no tumor)
])

# Compile the model
model.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the model
history = model.fit(
    train_gen, 
    epochs = EPOCHS, 
    validation_data = val_gen, 
    steps_per_epoch = train_gen.samples // BATCH_SIZE,
    validation_steps = val_gen.samples // BATCH_SIZE
)

# Save the trained model
if not os.path.exists(os.path.dirname(MODEL_PATH)):
    os.makedirs(os.path.dirname(MODEL_PATH))
model.save(MODEL_PATH)

print(f"Model saved at {MODEL_PATH}")
