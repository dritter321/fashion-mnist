from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.model import LitFashionMNIST


app = Flask(__name__)

df = pd.read_csv('../training/mlruns/run_logs.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
latest_row = df.loc[df['timestamp'].idxmax()]
experiment_id = latest_row['experiment_id']
run_id = latest_row['run_id']

model = LitFashionMNIST()
model.load_state_dict(torch.load(f"../training/mlruns/{experiment_id}/{run_id}/model_artifact/model.pth"))
model.eval()

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


@app.route('/infer', methods=['POST'])
def infer():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        # Read the image file
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        # Preprocess the image
        img_tensor = transform(img)

        # Add an extra batch dimension since pytorch treats all inputs as batches
        img_tensor = img_tensor.unsqueeze(0)

        # Predict the class from the image
        with torch.no_grad():
            prediction = model(img_tensor)
            _, predicted_idx = torch.max(prediction, 1)

        return jsonify({'predicted_class': predicted_idx.item()})

    return jsonify({'error': 'Unexpected error occurred'}), 500


if __name__ == '__main__':
    app.run(debug=True)