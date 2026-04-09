import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import joblib
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# paths
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'model/model.pkl'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# class names  order
class_names = [
    'Acne and Rosacea Photos',
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
    'Atopic Dermatitis Photos',
    'Bullous Disease Photos',
    'Cellulitis Impetigo and other Bacterial Infections',
    'Eczema Photos',
    'Exanthems and Drug Eruptions',
    'Hair Loss Photos Alopecia and other Hair Diseases',
    'Herpes HPV and other STDs Photos',
    'Light Diseases and Disorders of Pigmentation',
    'Lupus and other Connective Tissue diseases',
    'Melanoma Skin Cancer Nevi and Moles',
    'Nail Fungus and other Nail Disease',
    'Poison Ivy Photos and other Contact Dermatitis',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Scabies Lyme Disease and other Infestations and Bites',
    'Seborrheic Keratoses and other Benign Tumors',
    'Systemic Disease',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Urticaria Hives',
    'Vascular Tumors',
    'Vasculitis Photos',
    'Warts Molluscum and other Viral Infections'
]

# Loading the dictionary containing the model
model_dict = joblib.load(MODEL_PATH)

# Reconstructing the model
model_architecture = model_dict['architecture']
model_weights = model_dict['weights']
model = model_from_json(model_architecture)
model.set_weights(model_weights)


app.jinja_env.globals.update(zip=zip)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_img(img_path, img_size=(260, 260)):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img_path):
    img_array = preprocess_img(img_path)
    predictions = model.predict(img_array)[0]
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class] * 100
    return predicted_class, confidence, predictions

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # prediction
            predicted_class, confidence, predictions = predict_image(file_path)

            # predictions
            top_n = 5
            predictions_array = predictions.tolist()
            class_probabilities = list(zip(class_names, predictions_array))
            # Sort by probability
            class_probabilities.sort(key=lambda x: x[1], reverse=True)
            # top 5 predictions
            top_predictions = class_probabilities[:top_n]

            return render_template('index.html', filename=filename, prediction=class_names[predicted_class],
                                   confidence=confidence, predictions=predictions_array, class_names=class_names,
                                   top_predictions=top_predictions)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    # if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
