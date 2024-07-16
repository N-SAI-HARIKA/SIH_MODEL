from flask import Flask, render_template, request, jsonify, url_for

import os
import tensorflow as tf
import tensorflow.compat.v2 as tf
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.inception_v3 import preprocess_input

app = Flask(__name__)

# Load the trained model from the HDF5 file
model = tf.keras.models.load_model('model4.h5')
target_size = (299, 299)

def load_and_preprocess_new_image(file_path, target_size=target_size):
    img = load_img(file_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/')
def index():
    return render_template('index.html', uploaded_image=None, prediction_label=None, prediction_prob=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded file temporarily
    temp_path = 'static/temp_image.jpg'  # Save the file in the 'static' folder
    file.save(temp_path)

    # Load and preprocess the new image
    new_image = load_and_preprocess_new_image(temp_path)

    # Make predictions using the trained model
    prediction_result = model.predict(np.expand_dims(new_image, axis=0))

    # Interpret the prediction result
    prediction_label = "Dirt Buildup" if prediction_result[0][0] > 0.5 else "Clean"

    # Pass the image URL, prediction label, and probability to the template
    return render_template('index.html', uploaded_image=url_for('static', filename='temp_image.jpg'), prediction_label=prediction_label, prediction_prob=float(prediction_result[0][0]))

if __name__ == '__main__':
    app.run(debug=True)
