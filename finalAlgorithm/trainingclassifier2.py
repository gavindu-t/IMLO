import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
# loading CIFAR 10

# transform images in dataset to tensors of normalised range [-1,1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

# loading training set and test set
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 channel input, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(3, 15, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(15, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

learning_rate = 0.001
momentum = 0.9
epochs = 15

# loss function
criterion = nn.CrossEntropyLoss()
# gradient descent optimiser
optimiser = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)


def train_loop(trainloader, net, criterion, optimiser, epoch_number):
    net.train()
    running_loss = 0.0
    for batch, data in enumerate(trainloader):
        # get images in batch, labels is the answer to get
        images, labels = data
        # zero the parameter gradients
        optimiser.zero_grad()

        # Computes prediction and loss
        outputs = net(images)
        loss = criterion(outputs, labels)
        # Backpropogation and performs gradient descent step
        loss.backward()
        optimiser.step()

        # print statistics
        running_loss += loss.item()
        if batch % 2000 == 1999:    # print every 2000 mini-batches
            print(
                f'[{epoch_number + 1}, {(batch + 1) * 4:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0


def test_loop(trainloader, net, criterion):
    net.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    # removes calculation of gradients for outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            running_loss += criterion(outputs, labels).item()
            correct += (predicted == labels).sum().item()

            test_loss = running_loss / len(trainloader)
    print(
        f"Test Error: \n Accuracy: {(correct/100):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(trainloader, net, criterion, optimiser, t)
    test_loop(testloader, net, criterion)
print('Finished')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
