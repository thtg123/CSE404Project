import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from HandWritingDataset import HandWriting

from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
from IPython import display

batch_size = 32

dataset = HandWriting(csv_file='./Data/english.csv', root_dir='./Data',
                      transform=transforms.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset, [2728, 682])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

for X, y in test_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# model = torchvision.models.googlenet(pretrained=True)
# model.to(device)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()

        # Using the nn.Sequential function, define a 5 layers of convolutions
        self.conv_relu_stack = nn.Sequential(
            # with ReLU activation function
            # with sizes of input and output nodes as:
            # layer1: 1,32 , kernel size of 3, with padding of 1
            nn.MaxPool2d(16, 16),
            nn.Conv2d(3, 32, 3, padding=3),
            # layer2: 32, 64, kernel_size 3, with padding of 1
            nn.Conv2d(32, 64, 3, padding=3),
            # pooling layer
            nn.MaxPool2d(2, 2),
            # with ReLU activation function
            #nn.ReLU(),
            # you can add more conv, pooling and relu layers
            nn.Conv2d(64, 128, 3, padding=3),
            nn.MaxPool2d(2, 2),
            #nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=3),
            nn.MaxPool2d(2, 2),
            #nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=3),
            nn.MaxPool2d(2, 2),
            #nn.ReLU(),
            # Last layer: in:512, out: 10 (for 10 output classes)
            nn.Conv2d(512, 62, 3)
        )
        self.linear = nn.Linear(1860, 62)

    def forward(self, x):
        x = self.conv_relu_stack(x)
        x = self.flatten(x)
        logits = self.linear(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

# Use cross-entropy loss as the loss function
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-2

# Define a pytorch optimizer using stochastic gradient descent (SGD)
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

# Stabilize across runs
torch.manual_seed(42)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    # set the model to train mode
    model.train()

    losses = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)

        # Compute training loss
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()

        # Calculate model gradients from the loss and optimize the network
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        if batch % 100 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        losses.append(loss)

    return np.array(losses).mean()


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    # Set the model to eval mode
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():  # no_grad mode doesn't compute gradients
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)  # compute predictions from X
            test_loss += loss_fn(pred, y).item()  # compute the test loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # number of correct predictions
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct


epochs = 10

# for plotting the training loss
history = {'losses': [], 'accuracies': []}
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    history['losses'].append(train(train_loader, model, loss_fn, optimizer))
    history['accuracies'].append(test(test_loader, model, loss_fn))


print("Done!")



