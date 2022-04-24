import base64
import json
import requests

image_path = 'test.jpg'
api = 'http://localhost:5000/predict'

with open(image_path, 'rb') as f:
    img_bytes = f.read()
img_b64 = base64.b64encode(img_bytes).decode('utf8')

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

payload = json.dumps({"image": img_b64, "other_key": "value"})
response = requests.post(api, data=payload, headers=headers)
try:
    data = response.json()
    print(data)
except requests.exceptions.RequestException:
    print(response.text)
print('hi')
