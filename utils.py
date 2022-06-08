# imports
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import random

SEED = 2022
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def dataload():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = CIFAR10(root='data/',  download=True, train=True, transform=transform)
    test_dataset = CIFAR10(root='data/', download=True, train=False, transform=transform)
    
    train_size = 25000
    train_data, _ = random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    return train_loader, test_loader


def loadmodel():
    alexnet = models.alexnet(pretrained=True, dropout=0)
    alexnet.classifier[6] = nn.Linear(4096,10)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    alexnet.device = device

    return alexnet.to(device), device


def train(model, optimizer, criterion, train_loader, device):
    epoch = 1
    start_time = time.time()
    #Keep track of epoch accuracies and losses
    accuracies = []
    losses = []
    #Train until 100% accuracy
    while True:
        running_loss = 0.0
        num_correct = 0
        num_samples = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            #Get predictions
            output = model(inputs)
            loss = criterion(output, labels)
            #Optimise
            loss.backward()
            optimizer.step()

            #Keep track of correct predictions and number of samples to calculate accuracy later
            _, predictions = torch.max(output.data, 1)
            num_correct += (predictions == labels).sum().item()
            num_samples += predictions.size(0)

            #Update current epoch loss
            running_loss += loss.item()

        #Calculate epoch accuracy and average loss
        accuracy = float(num_correct) / float(num_samples) * 100
        accuracies.append(accuracy)
        losses.append(running_loss/len(train_loader))
        
        print(f"Epoch {epoch} accuracy: {accuracy:.2f}. Loss: {running_loss/len(train_loader):.3f}")

        #Finish training once accuracy is 100% or loss is 0
        if running_loss == 0 or accuracy == 100:
            print('Finished Training of AlexNet')
            print(f"Number of epochs until 100% accuracy: {epoch}")
            print(f"Time taken: {time.time() - start_time}")
            break
        epoch += 1
    return accuracies, losses
