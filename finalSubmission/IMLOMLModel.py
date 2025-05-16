import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import time


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(384 * 2 * 2, 768)
        self.fc2 = nn.Linear(768, 10)
        self.dropout = nn.Dropout(0.25)
        #
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

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # x = x.view(-1, 128*4*4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_loop(trainloader, net, criterion, optimiser, scheduler, batch_size, epoch_number):
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
        scheduler.step()
        # print statistics
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        batch_loss += loss.item()
        correct += (predicted == labels).sum().item()
        if batch % 50 == 49:    # print every 100 mini-batches
            print(
                f'[{epoch_number + 1}, {(batch + 1) * batch_size:5d}] loss: {batch_loss / 50:.3f}')
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


if __name__ == "__main__":
    # loading CIFAR 10
    start_time = time.time()
    # First randomly flip, rotate, crop and change colour values of image
    # This stops algorithm from overfitting to test data
    # Then transform images in dataset to tensors of normalised range [-1,1]
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
        transforms.RandomErasing(0.3, (0.02, 0.25))
    ])

    # Test images don't get augmented, just normalised
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2470, 0.2435, 0.2616])
    ])

    # 64 training examples in each pass.
    batch_size = 128

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
        num_workers=40)

    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=40)

    # Specify image classes
    classes = ('airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = Net()

    learning_rate = 0.02
    weight_decay = 0.003
    # momentum = 0.9

    avg_epoch_time = 160
    timeout = 60*60*3.94
    epochs = int(timeout/avg_epoch_time)

    print(f"Model expected to run for {epochs} epochs")

    best_epoch = 0
    best_test_loss = float("inf")
    best_test_acc = 0.0
    patience = 20
    patience_lost_counter = 0
    train_losses, train_accs, test_losses, test_accs = [], [], [], []

    PATH = './cifar_best_model10.pth'

    # loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Adam optimiser and learning rate scheduler
    optimiser = optim.AdamW(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=learning_rate, total_steps=epochs * len(trainloader))

    for epoch in range(epochs):
        epoch_start_time = time.time()

        print(f"Epoch {epoch+1}\n-------------------------------")
        current_lr = optimiser.param_groups[0]["lr"]
        print(f"Current Learning Rate: {current_lr}\n")

        train_loss, train_acc = train_loop(
            trainloader, net, criterion, optimiser, scheduler, batch_size, epoch)
        test_loss, test_acc = test_loop(testloader, net, criterion)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%\n")

        epoch_elapsed_time = time.time() - epoch_start_time
        total_elapsed_time = time.time() - start_time
        print(
            f"Epoch Run Time: {time.strftime('%M:%S', time.gmtime(epoch_elapsed_time))}")
        print(
            f"Total Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(total_elapsed_time))}")

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
            print(
                f"Early stopping counter: {patience_lost_counter}/{patience}")
            if patience_lost_counter >= patience:
                print("Early stopping triggered.")
                break
        if total_elapsed_time > timeout:
            print(f"Training time cap met.")
            break

    print('Finished Training')
    print("------------------")
    print(" MODEL STATISTICS ")
    print(f"Best Epoch: {best_epoch+1}")
    print(
        f"Time taken: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
    print("")
    print(f"Train Loss: {train_losses[best_epoch]}")
    print(f"Test Loss: {test_losses[best_epoch]}")
    print("")
    print(f"Train Accuracy: {train_accs[best_epoch]}")
    print(f"Test Accuracy: {test_accs[best_epoch]}")
    print("")
    print(f"Model was saved to {PATH}")
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
    plt.savefig("classifier10graph.png")

