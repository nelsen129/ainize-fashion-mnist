# Ainize Fashion MNIST

[![Run on Ainize](https://ainize.ai/images/run_on_ainize_button.svg)](https://ainize.ai/nelsen129/ainize-fashion-mnist?branch=main)

Fashion MNIST using the Ainizer platform

This is the backend server for the project. You can view the frontend repo by going [here](https://github.com/nelsen129/ainize-fashion-mnist-frontend)

## Dataset

You can find the dataset for Fashion MNIST [here](https://github.com/zalandoresearch/fashion-mnist)

## Initial setup

```bash
git clone https://github.com/nelsen120/ainize-fashion-mnist.git
cd ainize-fashion-mnist
python3 -m pip install tensorflow==2.6.* tensorflow-addons
```

## Backend server

### Setup

```bash
docker build --tag {project-name}:{tag} . 
docker run -p 5000:5000 {project-name}:{tag}
```

The backend server should now be running and you can interact with the model API!

### API 

#### Post Parameter

```
image: 28x28px grayscale image encoded to Base64 in UTF-8
```

This image should be loaded properly in order for the API call to work. 
Here's an example of proper loading in python

```python
from PIL import Image
import keras
img = Image.open(image_path)
img_arr = keras.utils.img_to_array(img)
img_arr = keras.preprocessing.image.smart_resize(img_arr, (28, 28))
img = keras.utils.array_to_img(img_arr)
img = img.convert('L')  # grayscale
img_bytes = img.tobytes()
img_b64 = base64.b64encode(img_bytes).decode('utf8')
```

Pass `img_b64` into the payload in the POST request

#### Output format

```python
{"prediction": Generated prediction}
```

This prediction is the category that the model predicts. The categories are T-shirt/top, trouser, pullover, dress, 
coat, sandal, shirt, sneaker, bag, and ankle boot
