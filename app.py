from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Initialize Flask app
app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = tf.keras.models.load_model('brain_tumor_cnn_model.h5')

# Image dimensions
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Prediction function
def predict_image(file_path):
    img = load_img(file_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = img_array.reshape(1, IMG_HEIGHT, IMG_WIDTH, 3)
    prediction = model.predict(img_array)
    return "Tumor Detected" if prediction[0] > 0.5 else "No Tumor"

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        result = predict_image(file_path)
        return render_template('result.html', result=result, image_path=file.filename)

if __name__ == '__main__':
    app.run(debug=True)
