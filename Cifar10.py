#####################
# Cifar10-Classifier
# Program that classifes digits, written in Python using PyTorch
# by Liviu Rotaru (c) 2018
#####################

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
parser.add_argument("--trbs", default = 3)
parser.add_argument("--tstbs", default = 4)

parser.add_argument("--lr", default = 0.001)
parser.add_argument("--mom", default = 0.9)
parser.add_argument("--epochs", default = 2)

args = parser.parse_args()





def loadData():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=20,
                                              shuffle=True, num_workers=2, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return transform, trainset, trainloader, testset, testloader, classes

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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.batchnorm = nn.BatchNorm2d(20)
        self.dropout = nn.Dropout2d(p = 0.4)
        self.fc1 = nn.Linear(20 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.batchnorm(x)
        x = x.view(-1, 20 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def main():

    net = Net()

    net.cuda()

    #transform, trainset, trainloader, testset, testloader, classes = loadData()
    dataM = DataManager()
    TrainM = TrainManager()


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = float(TrainM.lr), momentum = float(TrainM.momentum))

    correct = 0
    total = 0

    for epoch in range(int(TrainM.epochs)):  # loop over the dataset multiple times

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
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
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

