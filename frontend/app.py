import base64
import json
import requests
import streamlit as st

test_image_path = 'test.jpg'
api = 'https://main-ainize-fashion-mnist-nelsen129.endpoint.ainize.ai/predict'

categories = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']


def send_request(image_path):
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf8')

    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

    payload = json.dumps({"image": img_b64})
    response = requests.post(api, data=payload, headers=headers)
    status_code = response.status_code
    return status_code, response


st.title("Fashion-MNIST Classification")
st.header("Classify Fashion-MNIST using TensorFlow and Convolutional Neural Networks")

st.image(test_image_path)

if st.button("Submit"):
    status_code, response = send_request(test_image_path)
    if status_code == 200:
        prediction = response.json()
        max_score = max(prediction["prediction"])
        category = categories[[i for i in range(len(categories)) if prediction["prediction"][i] == max_score][0]]
        st.success(category)
    else:
        st.error(str(status_code) + " Error")

