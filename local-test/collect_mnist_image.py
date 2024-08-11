import torch
from torchvision import datasets, transforms
from PIL import Image

# Define the transform to convert data to tensor
transform = transforms.Compose([
    transforms.ToTensor()  # Convert image to tensor
])

# Download Fashion MNIST dataset
fashion_mnist_train = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Extract one image and its label
image_tensor, label = fashion_mnist_train[0]

# Convert to PIL Image and save to file
image = transforms.ToPILImage()(image_tensor.squeeze())
image.save("sample_fashion_mnist.png")

print(f"Image saved with label: {label}")