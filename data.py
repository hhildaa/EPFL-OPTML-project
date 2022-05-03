# imports
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision.models as models
import torchvision.transforms as transforms

def main():
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    # dataset
    dataset = CIFAR10(root='data/', download=True, transform=transform_train)
    test_dataset = CIFAR10(root='data/', train=False, transform=transform_test)

    classes = dataset.classes

    # # example data
    # img, label = dataset[0]
    # plt.imshow(img.permute((1, 2, 0)))
    # print('Label (numeric):', label)
    # print('Label (textual):', classes[label])

    torch.manual_seed(1)
    val_size = 5000
    train_size = len(dataset) - val_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    batch_size=3

    # train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # val_loader = DataLoader(val_data, batch_size*2, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size*2, num_workers=4, pin_memory=True)

    alexnet = models.alexnet(pretrained=True)

    num_correct = 0
    num_samples = 0
    for batch_idx, (data,targets) in enumerate(test_loader):
        data = data.to(device="cpu")
        targets = targets.to(device="cpu")
        ## Forward Pass
        scores = alexnet(data)
        _, predictions = scores.max(1)
        num_correct += (predictions == targets).sum()
        num_samples += predictions.size(0)
    print(
        f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
    )


if __name__ == '__main__':
    main()
