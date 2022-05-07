import torch
import numpy as np
import random
import data
import model
import train
import test

SEED = 2022

# set random seed for REPRODUCIBILITY
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def main():

    train_loader, val_loader, test_loader = data.dataload()

    alexnet = model.loadmodel()

    trained_model = train.train(alexnet, train_loader)

    test.test(trained_model, test_loader)

if __name__ == "__main__": 
    main()

