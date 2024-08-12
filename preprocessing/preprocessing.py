import yaml
from torchvision import datasets, transforms
import torch

# Load configuration
with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_config = config['data']
preprocessing_config = config['preprocessing']

# Define transformations using values from the config file
transform = transforms.Compose([
    transforms.Resize((preprocessing_config['image_size'], preprocessing_config['image_size'])),
    transforms.RandomHorizontalFlip(p=preprocessing_config['flip_probability']),
    transforms.ToTensor(),
    transforms.Normalize(
        preprocessing_config['normalization_params']['mean'],
        preprocessing_config['normalization_params']['std']
    )
])

# Load the raw datasets using the config paths
train_raw = datasets.FashionMNIST(root=data_config['raw_data_path'], train=True, download=True)
test_raw = datasets.FashionMNIST(root=data_config['raw_data_path'], train=False, download=True)

# Apply transformations and collect labels
train_transformed = []
train_labels = []

test_transformed = []
test_labels = []

for img, label in train_raw:
    train_transformed.append(transform(img))
    train_labels.append(label)

for img, label in test_raw:
    test_transformed.append(transform(img))
    test_labels.append(label)

# Save the preprocessed tensors and labels to disk using config paths
torch.save((train_transformed, train_labels), f"{data_config['preprocessed_data_path']}/{data_config['train_data_file']}")
torch.save((test_transformed, test_labels), f"{data_config['preprocessed_data_path']}/{data_config['test_data_file']}")