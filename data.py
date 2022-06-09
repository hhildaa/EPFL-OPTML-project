# imports
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms


def dataload(batch_size):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    dataset = CIFAR10(root='data/',  download=True, train=True, transform=transform)
    test_dataset = CIFAR10(root='data/', download=True, train=False, transform=transform)
    
    train_size = 25000
    train_data, _ = random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader