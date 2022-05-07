# imports
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms

def dataload():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = CIFAR10(root='data/',  download=True, train=True, transform=transform)
    test_dataset = CIFAR10(root='data/', download=True, train=False, transform=transform)

    val_size = 5000
    train_size = len(dataset) - val_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    batch_size=4

    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


