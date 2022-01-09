## https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

# Load MNIST dataset
training_data = datasets.MNIST(
    root="ai_digit_recognition/data", train=True, download=True, transform=ToTensor()
)

test_data = datasets.MNIST(
    root="ai_digit_recognition/data", train=False, download=True, transform=ToTensor()
)

# Map MNIST dataset
labels_map = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}

# Plt 3 x 3 sample
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
