"""
This is a convolutional neural network using PyTorch
that analyzes handwritten digits from the MNIST dataset
"""

import torch
from torch import nn  # Gets the neural network modules we need
from torch.utils.data import (
    DataLoader,
)
import torch.nn.functional as F
import torchvision.transforms as transforms  # transformations
from tqdm import tqdm
import torchvision.datasets as datasets # the regular datasets
from torch import optim  # For optimizers like SGD, Adam, etc.

# Convolutional Neural Network class
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__() # Initialize the NN using PyTorch nn module
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        ) # Defines the first layer of our NN

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # MaxPool to reduce the spatial dimensions, reducing parameters and making computational complexity lower
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        ) # Defines the 2nd layer of our NN

        self.fc1 = nn.Linear(16 * 7 * 7, num_classes) #First fully connected layer of NN

    def forward(self, x): # When the model calls the data and moves forward to the next layer
        x = F.relu(self.conv1(x)) # We use the Relu activation function
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x) # Pass through the fully connected layer
        return x


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Use gpu if we have it, else cpu

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 3e-4 # best learning rate we can use
batch_size = 64
num_epochs = 3

# Get the data
train_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root="dataset/", train=False, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#Initialize model to be the CNN we defined above
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Trains the network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

# Check accuracy on training set and test set
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")