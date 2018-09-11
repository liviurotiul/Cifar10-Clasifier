from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description = 'clasifier pe sifarten')
parser.add_argument("--trbs", default = 4)
parser.add_argument("--tstbs", default = 3)

parser.add_argument("--lr", default = 0.001)
parser.add_argument("--mom", default = 0.9)
parser.add_argument("--epochs", default = 2)
parser.add_argument("--wdecay", default = 0.0004)

args = parser.parse_args()






class DataManager():
    def __init__(self):


        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=int(args.trbs),
                                                  shuffle=True, num_workers=2)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=int(args.tstbs),
                                                 shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class TrainManager():
    def __init__(self):
        self.lr = args.lr
        self.momentum = args.mom
        self.epochs = args.epochs
        self.weight_decay = args.wdecay


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout = nn.Dropout2d()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.batchnorm5 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512 * 11 * 11, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)

        x = F.relu(self.conv3(x))
        x = self.batchnorm3(x)
        x = self.dropout(x)

        x = F.relu(self.conv4(x))
        x = self.batchnorm4(x)
        x = self.dropout(x)

        x = F.relu(self.conv5(x))
        x = self.batchnorm5(x)
        x = self.dropout(x)

        x = x.view(-1, 512 * 11 * 11)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():

    net = Net()

    net.cuda()

    dataM = DataManager()
    TrainM = TrainManager()


    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr = float(TrainM.lr), momentum = float(TrainM.momentum))
    optimizer = optim.Adam(net.parameters(), lr = float(TrainM.lr), weight_decay = float(TrainM.weight_decay))


    correct = 0
    total = 0

    for epoch in range(int(TrainM.epochs)):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        for i, data in enumerate(dataM.trainloader, 0):
        # get the inputs
            inputs, labels = data

            inputs, labels = inputs.cuda(), labels.cuda()
        # zero the parameter gradients
            optimizer.zero_grad()

        # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        with torch.no_grad():
            net.eval()
            for data in dataM.testloader:
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
        print('Finished epoch')
    print('Finished training')


    with torch.no_grad():
        for data in dataM.testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


if __name__ == '__main__':
   main()

