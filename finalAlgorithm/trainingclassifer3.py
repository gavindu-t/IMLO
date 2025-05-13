import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import hyperopt
from hyperopt import hp, fmin, tpe, Trials, space_eval, STATUS_OK
import time

start_time = time.time()

# First randomly flip, rotate, crop and change colour values of image
# This stops algorithm from overfitting to test data
# Then transform images in dataset to tensors of normalised range [-1,1]
train_transform = transforms.Compose([
    transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Test images don't get augmented, just normalised
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 64 training examples in each pass.
batch_size = 64

# loading training set and test set
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=train_transform)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=test_transform)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0)

# Specify image classes
classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Actual neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

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
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # x = x.view(-1, 128*4*4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()

learning_rate = 0.001
weight_decay = 0.0001
momentum = 0.9
epochs = 60

best_epoch = 0
best_test_loss = float("inf")
best_test_acc = 0.0
patience = 5
patience_lost_counter = 0
train_losses, train_accs, test_losses, test_accs = [], [], [], []

PATH = './cifar_best_model2.pth'

# loss function
criterion = nn.CrossEntropyLoss()
# Adam optimiser and learning rate scheduler
optimiser = optim.Adam(net.parameters(), lr=learning_rate,
                       weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, "min", patience=3, factor=0.5)


def train_loop(trainloader, net, criterion, optimiser, epoch_number):
    net.train()
    running_loss = 0.0
    batch_loss = 0.0
    correct = 0
    total = 0
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
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        batch_loss += loss.item()
        correct += (predicted == labels).sum().item()
        if batch % 100 == 99:    # print every 100 mini-batches
            print(
                f'[{epoch_number + 1}, {(batch + 1) * 64:5d}] loss: {batch_loss / 100:.3f}')
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
    # removes calculation of gradients for outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            loss = criterion(outputs, labels)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            running_loss += loss.item()
            correct += (predicted == labels).sum().item()

    test_loss = running_loss / len(testloader)
    test_accuracy = correct / total * 100
    return test_loss, test_accuracy


for epoch in range(epochs):
    epoch_start_time = time.time()
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_loss, train_acc = train_loop(
        trainloader, net, criterion, optimiser, epoch)
    test_loss, test_acc = test_loop(testloader, net, criterion)
    scheduler.step(test_loss)
    print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

    epoch_elapsed_time = time.time() - epoch_start_time
    print(
        f"Epoch Run Time: {time.strftime(' % H: % M: % S', time.gmtime(epoch_elapsed_time))}")

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    # Early stopping logic
    if test_acc > best_test_acc:
        best_epoch = epoch
        best_test_acc = test_acc
        best_test_loss = test_loss
        patience_lost_counter = 0
        torch.save(net.state_dict(), PATH)
        print("Best model saved: Accuracy has been increased")
    else:
        patience_lost_counter += 1
        print(f"Early stopping counter: {patience_lost_counter}/{patience}")
        if patience_lost_counter >= patience:
            print("Early stopping triggered.")
            break
print('Finished Training')
print("------------------")
print(" MODEL STATISTICS ")
print(f"Best Epoch: {best_epoch}")
print("")
print(f"Train Loss: {train_losses[best_epoch]}")
print(f"Test Loss: {test_losses[best_epoch]}")
print("")
print(f"Train Accuracy: {train_accs[best_epoch]}")
print(f"Test Accuracy: {test_accs[best_epoch]}")

total_elapsed_time = time.time() - start_time
print(f"Total Run Time: {time.strftime(" % H: % M: % S", time.gmtime(total_elapsed_time))}")
print("------------------")


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
