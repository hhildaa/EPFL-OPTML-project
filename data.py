# imports
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
import numpy as np

from scipy.stats import truncnorm


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
    #Resamples the data to be a truncated Gaussian of min 0 and max 255.
    #Gaussian needs to be truncated to be withing [0,255] or else PIL complains
    def random_images(self):
        #Generate initial sample
        np.random.seed(2022)
        mu, sigma = np.mean(self.data), np.std(self.data)
        size = np.prod(self.data.shape)
        X = np.random.normal(mu, sigma, size)

        #Check if any are out of range
        lt_ind = X < 0
        gt_ind = X > 255
        total = np.sum(lt_ind) + np.sum(gt_ind)
        #If they are, keep resampling until all values are within the range
        #This is a dumb approach, but much faster than scipy.stats.truncnorm (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html)
        #for some reason (on Google Colab)
        while total != 0:
            np.random.seed(2022)
            new = np.random.normal(mu, sigma, total)
            X[lt_ind | gt_ind] = new
            
            lt_ind = X < 0
            gt_ind = X > 255
            total = np.sum(lt_ind) + np.sum(gt_ind)
            
        self.data = X.reshape(self.data.shape).astype('uint8')

def dataload(batch_size=64, corrupt_prob=0, perm_level=0, random_noise=False):
    """
    Loading the data into train and testloader.
    ------
    batch_size: Batch size.
    corrupt_prob: Probability of corrupting the labels. Number in [0,1].
    perm_level: Takes value in {0,1,2}. 0: means no permutation of pixels. 1: permutation of pixels 
                over x-axis, 2: permutation over x- and y-axis.
    random_noise: Boolean, decides on whether giving Gaussian noise to the pixels or not.
    """
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