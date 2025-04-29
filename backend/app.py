# final_year_proj/backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the model from the parent directory
model = tf.keras.models.load_model('../best_model_transfer_learning.h5')

def preprocess(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    img_file = request.files['file']
    os.makedirs('uploads', exist_ok=True)
    file_path = os.path.join('uploads', img_file.filename)
    img_file.save(file_path)

    img_array = preprocess(file_path)
    prediction = model.predict(img_array)[0][0]
    label = "Non-Dangerous Animal" if prediction > 0.5 else "Dangerous Animal"
    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(debug=True)
