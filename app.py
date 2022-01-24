from flask import Flask, request, render_template, jsonify
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import base64
import numpy as np
import io
from PIL import Image

app = Flask(__name__)


def get_model():
    global model
    # model = load_model("model.kerasmodel")
    model = load_model("model.h5")
    print(" * Model loaded!")


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


print(" * Loading Keras model...")
get_model()


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('predict.html')


@app.route("/predict", methods=["POST"])
def predict():
    # return render_template('predict.html')
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224, 224))

    prediction = model.predict(processed_image).tolist()

    response = {
        'prediction': {
            'cat': prediction[0][0],
            'dog': prediction[0][1]
        }
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run()
