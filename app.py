from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('dog_cat_classifier_model.h5')

# Define image size (same as in training)
IMG_SIZE = 128

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling the image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        # Save the uploaded file
        filepath = os.path.join('static', file.filename)
        file.save(filepath)

        # Preprocess the image for the model
        img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        prediction = model.predict(img_array)

        # Determine if the image is of a dog or cat
        if prediction[0][0] > 0.5:
            result = f"Dog (Confidence: {prediction[0][0]:.2f})"
        else:
            result = f"Cat (Confidence: {1 - prediction[0][0]:.2f})"
        
        return render_template('result.html', result=result, img_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
