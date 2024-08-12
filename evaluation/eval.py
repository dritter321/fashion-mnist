import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.model import LitFashionMNIST
import pandas as pd
import torchmetrics

# Load eval dataset
eval_data, eval_labels = torch.load('../data/test_preprocessed.pt')
eval_data = torch.stack(eval_data)
eval_labels = torch.tensor(eval_labels)

# Create dataset and dataloader
eval_dataset = TensorDataset(eval_data, eval_labels)
eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

# Load the trained model
model = LitFashionMNIST()
df = pd.read_csv('../training/mlruns/run_logs.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
latest_row = df.loc[df['timestamp'].idxmax()]
experiment_id = latest_row['experiment_id']
run_id = latest_row['run_id']
model_path = f"../training/mlruns/{experiment_id}/{run_id}/model_artifact/model.pth"
model.load_state_dict(torch.load(model_path))
model.eval()

# Set up metrics for multiclass classification
accuracy = torchmetrics.Accuracy(num_classes=10, average='macro', task='multiclass')
precision = torchmetrics.Precision(num_classes=10, average='macro', task='multiclass')
recall = torchmetrics.Recall(num_classes=10, average='macro', task='multiclass')
f1 = torchmetrics.F1Score(num_classes=10, average='macro', task='multiclass')  # Updated from F1 to F1Score

# Prepare device and send model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluate the model
results = defaultdict(list)
model.eval()
with torch.no_grad():
    for batch in eval_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)

        # Calculate metrics
        results['accuracy'].append(accuracy(preds, y).item())
        results['precision'].append(precision(preds, y).item())
        results['recall'].append(recall(preds, y).item())
        results['f1_score'].append(f1(preds, y).item())

# Average results across batches
for key in results:
    results[key] = sum(results[key]) / len(results[key])

# Log results to a CSV file
results_df = pd.DataFrame([results])
csv_file_path = './eval_results.csv'
file_exists = os.path.isfile(csv_file_path)
results_df.to_csv(csv_file_path, mode='a', header=not file_exists, index=False)

print("Evaluation results saved to", csv_file_path)