from flask import Flask, render_template, request
import keras
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# ==============================
# Load Model (Keras 3 compatible)
# ==============================
model = keras.models.load_model("model_35.keras", compile=False)

# ==============================
# Class Labels
# ==============================
verbose_name = {
    0: 'Carcinoma In Situ (SCCSI)',
    1: 'Mild Dysplasia (MS-NKD)',
    2: 'Moderate Dysplasia (MOS-NKD)',
    3: 'Columnar (CE)',
    4: 'Intermediate Squamous (ISE)',
    5: 'Superficial Squamous (SSE)',
    6: 'Severe Dysplasia (SS-NKD)'
}

# ==============================
# Prediction Function
# ==============================
def predict_label(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return verbose_name[predicted_class]


# ==============================
# Routes
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    if "my_image" not in request.files:
        return "No file uploaded"

    file = request.files["my_image"]

    if file.filename == "":
        return "No selected file"

    upload_folder = "static/tests"
    os.makedirs(upload_folder, exist_ok=True)

    img_path = os.path.join(upload_folder, file.filename)
    file.save(img_path)

    prediction = predict_label(img_path)

    return render_template("prediction.html",
                           prediction=prediction,
                           img_path=img_path)


if __name__ == "__main__":
    app.run(debug=True)