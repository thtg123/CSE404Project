import pandas as pd
import skimage
from PIL import Image
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, \
    Dropout
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, SubsetRandomSampler
from torchvision import transforms
import dataset

dataset = dataset.CustomDataset()
batch_size = 16
validation_split = .2
shuffle_dataset = True
random_seed = 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)
"""
data = pd.read_csv("./Data/english.csv")
labels = data.iloc[:, 1]
files = data.iloc[:, 0]

img_flats = []
img_norm = []

for i in files:
    img = np.array(np.asarray(Image.open("./Data/" + i))[:, :, 0])

    small_img = skimage.measure.block_reduce(img, (100, 100), np.min)
    img_norm.append(small_img)

    # flatten small array
    flat = []
    for x in small_img:
        for y in x:
            flat.append(y)
    for e in range(len(flat)):
        flat[e] = 1 if flat[e] != 0 else 0

    img_flats.append(flat)

    print(i)

img_norm = np.array(img_norm)
labels = np.array(labels)

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(img_norm, labels, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

X_train = X_train.reshape(X_train.shape[0], 9, 12)
trans = transforms.Compose([transforms.ToTensor()])
print(X_train.shape)
X_train = trans(X_train)
X_train = X_train.view(12, 1, 2046, 9)
#X_train = torch.from_numpy(X_train)
print(X_train.shape)

le = preprocessing.LabelEncoder()
targets = le.fit_transform(y_train)
y_train = torch.as_tensor(targets)
print(y_train.shape)

size = X_val.shape[0]
X_val = X_val.reshape(size, 9, 12)
X_val = trans(X_val)
X_val = X_val.view(12, 1, size, 9)
#X_val = torch.from_numpy(X_val)

le = preprocessing.LabelEncoder()
targets = le.fit_transform(y_val)
y_val = torch.as_tensor(targets)
"""


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(96, 63)
        )
        self.softMax = Sequential(
            Softmax(dim=1)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = self.softMax(x)
        return x


# defining the model
model = Net()
# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.07)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

print(model)

# Use cross-entropy loss as the loss function
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1e-2

# Define a pytorch optimizer using stochastic gradient descent (SGD)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# defining the number of epochs
n_epochs = 25
# empty list to store training losses
train_losses = []
# empty list to store validation losses
val_losses = []
# training the model
import numpy as np


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.data)

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
    size = len(dataloader.data)
    num_batches = len(dataloader)

    # Set the model to eval mode
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():  # no_grad mode doesn't compute gradients
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)  # compute predictions from X
            test_loss += loss_fn(pred, y.long()).item()  # compute the test loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # number of correct predictions
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct


for X, y in train_loader:
    print(type(X))
    print(type(y))
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

epochs = 10

# for plotting the training loss
history = {'losses': [], 'accuracies': []}
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    history['losses'].append(train(train_loader, model, loss_fn, optimizer))
    history['accuracies'].append(test(validation_loader, model, loss_fn))