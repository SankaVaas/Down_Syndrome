# app.py

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('./down_syn.ipynb')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        # Handle uploaded image
        if 'file' in request.files:
            image = request.files['file']
            if image.filename != '':
                # Preprocess the uploaded image
                image = preprocess_image(image)
                # Make a prediction
                prediction = model.predict(image)
                result = 'Down Syndrome' if prediction > 0.5 else 'Healthy'

    return render_template('index.html', result=result)

def preprocess_image(image):
    # Load and preprocess the image
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (250, 250))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

if __name__ == '__main__':
    app.run(debug=True)
