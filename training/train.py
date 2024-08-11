import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model import LitFashionMNIST
import mlflow.pytorch
import time
import os
from pathlib import Path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import log_run_details_to_file

experiment_name = "fashion_mnist_experiment"

### Loading data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

### Setting up logging
start = time.time()
mlflow_logger = MLFlowLogger(
    experiment_name=experiment_name,
    tracking_uri=mlflow.get_tracking_uri()
)
experiment_id = mlflow_logger.experiment_id
run_id = mlflow_logger.run_id
run_dir = f"./mlruns/{experiment_id}/{run_id}/"


### Starting model training
with mlflow.start_run() as run:
    model = LitFashionMNIST()
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        min_delta=0.00,
                                        patience=3,
                                        verbose=True,
                                        mode='min')

    checkpoint_callback = ModelCheckpoint(
        dirpath= run_dir + "model_checkpoints",
        filename='lit_fashion_mnist_model_{epoch:02d}_{val_loss:.2f}',
        save_top_k=-1, # Save all models
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=mlflow_logger
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