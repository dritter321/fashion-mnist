import pytorch_lightning as pl
import torch
import yaml

# Load configuration
with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

model_config = config['model']

### Defining training
class LitFashionMNIST(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.fc1 = torch.nn.Linear(model_config['input_size'], model_config['first_layer_output'])
        self.fc2 = torch.nn.Linear(model_config['first_layer_output'], model_config['second_layer_output'])
        self.fc3 = torch.nn.Linear(model_config['second_layer_output'], model_config['output_classes'])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.CrossEntropyLoss()(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.CrossEntropyLoss()(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=model_config['learning_rate'])