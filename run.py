import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import data
import model
import train
import test

SEED = 2022
LEARNING_RATE = 0.00001
BATCH_SIZE = 64
CORRUPT_PROB = [0, 0.2, 0.4, 0.6, 0.8, 1]
#EPOCH = 3

# set random seed for REPRODUCIBILITY
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def main():

    for corrupt_prob in CORRUPT_PROB: 
        print ("------------------------")
        print(f"Corruption probability: {corrupt_prob}")
        train_loader, _ = data.dataload(batch_size=BATCH_SIZE, corrupt_prob=corrupt_prob)

        alexnet, device = model.loadmodel()

        criterion = nn.CrossEntropyLoss()

        #rms_optimizer = optim.RMSprop(alexnet.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        #rms_accuracies, rms_losses = train.train(alexnet, rms_optimizer, criterion, train_loader, device)

        adam_optimizer = optim.Adam(alexnet.parameters(), lr=LEARNING_RATE)
        _,_ = train.train(alexnet, adam_optimizer, criterion, train_loader, device)

if __name__ == "__main__": 
    main()

