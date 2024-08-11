import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import pandas as pd

## Inference of random input on latest model

class LitFashionMNIST(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = LitFashionMNIST()

df = pd.read_csv('../training/mlruns/run_logs.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
latest_row = df.loc[df['timestamp'].idxmax()]
experiment_id = latest_row['experiment_id']
run_id = latest_row['run_id']

model.load_state_dict(torch.load(f"../training/mlruns/{experiment_id}/{run_id}/model_artifact/model.pth"))
model.eval()
test_dataset = datasets.FashionMNIST(root='./data',
                                     train=False,
                                     download=True,
                                     transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset,
                         batch_size=1,
                         shuffle=True)
images, labels = next(iter(test_loader))
transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
image = transform(images)

with torch.no_grad():
    logits = model(image)
    predictions = torch.argmax(logits, dim=1)
    predicted_label = test_dataset.classes[predictions.item()]

plt.imshow(images[0][0], cmap='gray')
plt.title(f'Predicted: {predicted_label}')
plt.colorbar()
plt.show()

print(f"Predicted class ID: {predictions.item()}, Label: {predicted_label}")