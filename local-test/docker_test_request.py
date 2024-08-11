import requests
import os

url = 'http://localhost:5000/infer'
dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(dir_path, 'sample_fashion_mnist.png')
files = {'file': open(file_path, 'rb')}
response = requests.post(url, files=files)
print(response.text)