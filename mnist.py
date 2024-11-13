# Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the Multilayer Perceptron (MLP) Model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Define the layers
        self.hidden1 = nn.Linear(28 * 28, 128)  # First hidden layer (784 inputs, 128 outputs)
        self.hidden2 = nn.Linear(128, 64)       # Second hidden layer (128 inputs, 64 outputs)
        self.output = nn.Linear(64, 10)         # Output layer (64 inputs, 10 outputs for each digit)

    def forward(self, x):
        # Flatten the input (28x28) to a vector of 784 elements
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.hidden1(x))         # Apply ReLU to the output of the first hidden layer
        x = torch.relu(self.hidden2(x))         # Apply ReLU to the output of the second hidden layer
        x = self.output(x)                      # Output layer (no activation here, CrossEntropy handles it)
        return x

# Initialize the model
model = MLP()

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()          # Loss function for multi-class classification
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # SGD optimizer with momentum

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# Train the model
def train(model, train_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()                 # Reset gradients
            outputs = model(images)               # Forward pass
            loss = criterion(outputs, labels)     # Calculate loss
            loss.backward()                       # Backward pass
            optimizer.step()                      # Update weights

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Train for 5 epochs
train(model, train_loader, criterion, optimizer, epochs=5)

# Evaluate the model's accuracy
def evaluate(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():  # No need to calculate gradients during evaluation
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = (correct / total) * 100
    print(f"Accuracy: {accuracy:.2f}%")

# Run evaluation
evaluate(model, test_loader)
