import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import time


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Fully connected layers, 2048 -> 1024 -> 256 -> 10
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)
        # 4 convolution blocks with 2, 2, 3, 3 convolution layers and max pooling
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = torch.flatten(x, 1)
        # Dropout to prevent overfitting
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_loop(trainloader, net, criterion, batch_size,):
    net.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    batch_loss = 0.0
    # Removes calculation of gradients for outputs
    with torch.no_grad():
        for batch, data in enumerate(trainloader):
            images, labels = data
            # Calculate outputs by running images through the network
            outputs = net(images)
            loss = criterion(outputs, labels)
            # The class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            running_loss += loss.item()
            correct += (predicted == labels).sum().item()
        if batch % 50 == 49:    # print every 50 mini-batches
            print(
                f'[, {(batch + 1) * batch_size:5d}] loss: {batch_loss / 50:.3f}')
            running_loss += batch_loss
            batch_loss = 0.0

    running_loss += batch_loss
    train_loss = running_loss / len(trainloader)
    train_accuracy = correct / total * 100
    return train_loss, train_accuracy


def test_loop(testloader, net, criterion):
    net.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    # Removes calculation of gradients for outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # Calculate outputs by running images through the network
            outputs = net(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            running_loss += loss.item()
            correct += (predicted == labels).sum().item()

    test_loss = running_loss / len(testloader)
    test_accuracy = correct / total * 100
    return test_loss, test_accuracy


if __name__ == "__main__":
    # loading CIFAR 10
    start_time = time.time()

    # Applies a variety of transforms to the images in order to artificially enlarge amount of images
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        # Normalised with values of mean and STD of CIFAR10
        # Values copied from stack overflow:
        # https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2470, 0.2435, 0.2616]),
        transforms.RandomErasing(0.35, (0.02, 0.25))
    ])

    # Test images don't get augmented, just normalised
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2470, 0.2435, 0.2616])
    ])

    batch_size = 256

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=30)

    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=30)

    # Specify image classes
    classes = ('airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    PATH = 'cifar_best_model16.pth'
    net = Net()
    # Removing prefixes from keys that came from compiled model
    loaded_dict = torch.load(PATH, weights_only=True)
    new_dict = {}
    for k, v in loaded_dict.items():
        if k.startswith("_orig_mod."):
            name = k[len("_orig_mod."):]  # Removes prefix
            new_dict[name] = v
    load_dict = new_dict
    net.load_state_dict(load_dict)
    net.eval()

    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    train_loss, train_acc = train_loop(
        trainloader, net, criterion, batch_size,)
    test_loss, test_acc = test_loop(testloader, net, criterion)
    # Final loss and accuracy
    print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%\n")
