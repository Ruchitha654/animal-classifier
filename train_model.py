import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from tensorflow.keras.preprocessing import image

# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

# Custom classification layers
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define dataset paths and generators
dataset_path = "dataset"
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Save the best model
checkpoint = ModelCheckpoint('best_model_transfer_learning.h5', monitor='val_accuracy', save_best_only=True)

# Train the model
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5,
    callbacks=[checkpoint]
)

# Classify Uploaded Image
def classify_animal(image_path):
    # Load the trained model
    trained_model = tf.keras.models.load_model('best_model_transfer_learning.h5')
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = trained_model.predict(img_array)[0][0]
    if prediction > 0.5:
        return "Non-Dangerous Animal"
    else:
        return "Dangerous Animal"

# Example Usage:
image_path = "catt.jpg"  # Replace with actual image path
result = classify_animal(image_path)
print(f"Prediction: {result}")
