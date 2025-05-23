import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import time
import sys

# NOTE:
# This model was trained on the departmental compute server csgpu1.cs.york.ac.uk
# It was also trained at a time where nobody else was using said server (shown by running htop)
# Hence this model was produced to utilise the full system resources available
# such as using torch.set_num_threads(35) to use 35 threads for processing instead of the initial 20
# The avg_epoch_time is correct at the time of training


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


def train_loop(trainloader, net, criterion, optimiser, scheduler, batch_size, epoch_number):
    net.train()
    running_loss = 0.0
    batch_loss = 0.0
    correct = 0
    total = 0
    for batch, data in enumerate(trainloader):
        # Get images in batch, labels is the answer to get
        images, labels = data
        # Zero the parameter gradients
        optimiser.zero_grad()
        # Computes prediction and loss
        outputs = net(images)
        loss = criterion(outputs, labels)
        # Backpropogation and performs gradient descent step
        loss.backward()
        optimiser.step()
        # Changes learning rate based on OneCycleLR
        scheduler.step()
        # Print statistics
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        batch_loss += loss.item()
        correct += (predicted == labels).sum().item()
        if batch % 50 == 4:    # print every 50 mini-batches
            print(
                f'[{epoch_number + 1}, {(batch + 1) * batch_size:5d}] loss: {batch_loss / 50:.3f}')
            running_loss += batch_loss
            batch_loss = 0.0

    running_loss += batch_loss
    train_loss = running_loss / len(trainloader)
    train_accuracy = correct / total * 100
    return train_loss, train_accuracy


if __name__ == "__main__":
    net = Net()
    if "linux" in sys.platform:
        torch.set_num_threads(35)
        net = torch.compile(net)

    # Applies a variety of transforms to prevent overfitting
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
        num_workers=30,
        persistent_workers=True)

    classes = ('airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    learning_rate = 0.02
    weight_decay = 0.003

    avg_epoch_time = 150
    timeout = 60*60*3.94
    epochs = int(timeout/avg_epoch_time)

    print(f"Model expected to run for {epochs} epochs")

    best_epoch = 0
    best_train_acc = 0.0
    train_losses, train_accs = [], []

    PATH = './cifar_best_model17.pth'

    # loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Adam optimiser and OneCycleLR learning rate scheduler
    optimiser = optim.AdamW(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=learning_rate, total_steps=epochs * len(trainloader))

    start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()

        print(f"Epoch {epoch+1}\n-------------------------------")
        current_lr = optimiser.param_groups[0]["lr"]
        print(f"Current Learning Rate: {current_lr}\n")

        train_loss, train_acc = train_loop(
            trainloader, net, criterion, optimiser, scheduler, batch_size, epoch)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

        epoch_elapsed_time = time.time() - epoch_start_time
        total_elapsed_time = time.time() - start_time
        print(
            f"Epoch Run Time: {time.strftime('%M:%S', time.gmtime(epoch_elapsed_time))}")
        print(
            f"Total Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(total_elapsed_time))}")

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        torch.save(net.state_dict(), PATH)
        if train_acc > best_train_acc:
            best_epoch = epoch

    print('Finished Training')
    print("------------------")
    print(" MODEL STATISTICS ")
    print(f"Best Epoch: {best_epoch+1}")
    print(
        f"Time taken: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
    print("")
    print(f"Train Loss: {train_losses[best_epoch]}")
    print("")
    print(f"Train Accuracy: {train_accs[best_epoch]}")
    print("")
    print(f"Model was saved to {PATH}")
    print("------------------")
