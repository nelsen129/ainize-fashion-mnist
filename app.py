import base64
import tensorflow.keras as keras

from flask import Flask, request, jsonify, abort
from PIL import Image

categories = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

app = Flask(__name__)

model = keras.models.load_model('./Fashion-MNIST/model.h5')


def predict_class(image):
    try:
        # Generate prediction
        outputs = model(image[None, :, :])

        prediction = outputs[0].numpy().tolist()
        max_score = max(prediction)
        category = categories[[i for i in range(len(categories)) if prediction[i] == max_score][0]]
        result = {"prediction": category}

        return jsonify(result)

    except Exception as e:
        print('Error occur in image classification!', e)
        return jsonify({'error': e}), 400


@app.route("/predict", methods=["POST"])
def main():
    # Load image as bytes, decode to TF tensor
    try:
        img_b64 = request.form.get('image')
        img_bytes = base64.b64decode((img_b64.encode('utf-8')))
        img = Image.frombytes('L', (28, 28), img_bytes)
        img_arr = keras.utils.img_to_array(img)
        img_arr = keras.preprocessing.image.smart_resize(img_arr, (28, 28))  # Should be unnecessary, but just to be safe

        prediction = predict_class(img_arr)

    except Exception as e:
        return jsonify({'message': f'Invalid request, {e}'}), 400

    return prediction


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
