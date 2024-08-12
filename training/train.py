import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
import time
import os
from pathlib import Path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import log_run_details_to_file

# Load configuration
with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_config = config['data']
training_config = config['training']
logging_config = config['logging']

experiment_name = logging_config['experiment_name']

# Load the preprocessed data and labels
train_data, train_labels = torch.load(f"{data_config['preprocessed_data_path']}/{data_config['train_data_file']}")
test_data, test_labels = torch.load(f"{data_config['preprocessed_data_path']}/{data_config['test_data_file']}")

# Convert train_data and test_data from list of tensors to a single 3D Tensor
train_data = torch.stack(train_data)
test_data = torch.stack(test_data)

# Convert labels to tensors (if not already tensors)
train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)

# Create TensorDataset
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)

# Create DataLoader with the batch size from config
train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=training_config['shuffle_data'])
test_loader = DataLoader(test_dataset, batch_size=training_config['batch_size'])

### Setting up logging
start = time.time()
mlflow_logger = MLFlowLogger(
    experiment_name=experiment_name,
    tracking_uri=mlflow.get_tracking_uri()
)
run_id = mlflow_logger.run_id
experiment_id = mlflow_logger.experiment_id
run_dir = f"{logging_config['run_logs_dir']}/{experiment_id}/{run_id}/"

### Starting model training
with mlflow.start_run() as run:
    from model import LitFashionMNIST
    model = LitFashionMNIST()
    early_stop_callback = EarlyStopping(**training_config['early_stopping'])

    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir + logging_config['model_checkpoint_dir'],
        filename='lit_fashion_mnist_model_{epoch:02d}_{val_loss:.2f}',
        **training_config['checkpoint']
    )

    trainer = Trainer(
        max_epochs=training_config['max_epochs'],
        accelerator=training_config['accelerator'],
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=mlflow_logger,
        precision=training_config['precision'],
    )

    trainer.fit(model, train_loader, test_loader)
    trainer.test(model, dataloaders=test_loader)

    # Save model at the end of training to mlruns
    model_dir = Path(run_dir + "model_artifact")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pth"
    torch.save(model.state_dict(), model_path)

    log_run_details_to_file(experiment_name, experiment_id, run_id, time.time() - start, model_path)
    print("Model logs, checkpoints, and artifact stored to ./mlruns.")