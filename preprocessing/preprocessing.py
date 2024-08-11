from torchvision import datasets, transforms
import torch

# Define transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize images to 28x28
    transforms.RandomHorizontalFlip(),  # Optionally flip the images horizontally
    transforms.ToTensor(),  # Convert images to tensor format
    transforms.Normalize((0.5,), (0.5,))  # Normalize the tensor values
])

# Load the raw datasets
train_raw = datasets.FashionMNIST(root='../data', train=True, download=True)
test_raw = datasets.FashionMNIST(root='../data', train=False, download=True)

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

# Save the preprocessed tensors and labels to disk
torch.save((train_transformed, train_labels), '../data/train_preprocessed.pt')
torch.save((test_transformed, test_labels), '../data/test_preprocessed.pt')