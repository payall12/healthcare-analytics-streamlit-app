# ==========================================
# PNEUMONIA CNN MODEL TRAINING
# ==========================================
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import os

# Standardize image dimensions
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

print("Setting up Image Generators...")
# 1. Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Test data ONLY rescaled
test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from local data folder
train_generator = train_datagen.flow_from_directory(
    "Data/chest-xray-pneumonia/chest_xray/train",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_generator = test_datagen.flow_from_directory(
    "Data/chest-xray-pneumonia/chest_xray/train",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("Building CNN Architecture...")
# 2. Build the CNN
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training CNN Model... (This will take a while)")
# 3. Train Model
history = cnn_model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# 4. Save the Model
os.makedirs('models', exist_ok=True)
cnn_model.save("models/pneumonia_cnn_model.h5")
print("✅ CNN Model saved successfully in models/ folder!")