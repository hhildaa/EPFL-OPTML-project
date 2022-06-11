# imports
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
import numpy as np

#CIFAR10 random labelling taken from https://github.com/pluskid/fitting-random-labels
class CIFAR10RandomLabels(CIFAR10):
    """CIFAR10 dataset, with support for randomly corrupt labels.
    Params
    ------
    corrupt_prob: float
        Default 0.0. The probability of a label being replaced with
        random label.
    num_classes: int
        Default 10. The number of classes in the dataset.
    """
    def __init__(self, corrupt_prob=0.0, perm_level=0, random_noise=False, num_classes=10, **kwargs):
        super(CIFAR10RandomLabels, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)
        if perm_level > 0:
            self.permute_images(perm_level)
        if random_noise:
            self.random_images()

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]

        self.targets = labels

    def permute_pixels(self, image, perm_level):
        np.random.seed(12345)
        image = np.random.permutation(image)
        if perm_level==2:
            np.random.seed(12345)
            image = np.random.permutation(np.swapaxes(image, 1,0))
            image = np.swapaxes(image,1,0)
        return image
    def permute_images(self, perm_level):
        for i, image in enumerate(self.data):
            self.data[i] = self.permute_pixels(image, perm_level)

    def random_images(self):
        mean = np.mean(self.data)
        std = np.std(self.data)
        self.data = np.random.normal(mean, std, self.data.shape)

def dataload(batch_size=64, corrupt_prob=0, perm_level=0, random_noise=False):
    SEED = 2022
    torch.manual_seed(SEED)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Slightly improves performance, since Alexnet was trained on this
    ])

    dataset = CIFAR10RandomLabels(root='data/',  download=True, train=True, transform=transform, corrupt_prob=corrupt_prob, perm_level=perm_level, random_noise=random_noise)
    test_dataset = CIFAR10(root='data/', download=True, train=False, transform=transform)
    
    train_size = 25000
    train_data, _ = random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader