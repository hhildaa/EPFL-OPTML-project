from statistics import mode
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import data
import model
import train
import test

# PARAMETERS
LEARNING_RATE = 0.001
LEARNING_RATE_ADAM = 0.00001
BATCH_SIZE = 64

# RUNNING PARAMETERS
CORRUPT_PROB = [0]
NOISE = [False, True]
PERM_LEVEL = [1, 2]
OPTIMIZER = ["adam"]            # possible values: "adam", "sgd", "rmsprop"

# set random seed for REPRODUCIBILITY
SEED = 2022
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def main():

    for corrupt_prob in CORRUPT_PROB: 
        print ("------------------------")
        print(f"Corruption probability: {corrupt_prob}")

        if corrupt_prob == 0: 
            for noise in NOISE: 
                print(f"Noise: {noise}")

                train_loader, _ = data.dataload(batch_size=BATCH_SIZE, corrupt_prob=corrupt_prob, perm_level=0, random_noise=noise)
                criterion = nn.CrossEntropyLoss()

                for opt in OPTIMIZER: 
                    if opt == "rmsprop": 
                        print(f"Model: RMS")
                        alexnet, device = model.loadmodel()
                        rms_optimizer = optim.RMSprop(alexnet.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
                        rms_accuracies, rms_losses = train.train(alexnet, rms_optimizer, criterion, train_loader, device)
                    if opt == "adam":
                        print("Model: Adam")
                        alexnet, device = model.loadmodel()
                        adam_optimizer = optim.Adam(alexnet.parameters(), lr=LEARNING_RATE_ADAM)
                        _,_ = train.train(alexnet, adam_optimizer, criterion, train_loader, device) 
                    if opt == "sgd": 
                        print("Model: SGD")
                        alexnet, device = model.loadmodel()
                        sgd_optimizer = optim.SGD(alexnet.parameters(), lr=LEARNING_RATE)
                        _,_ = train.train(alexnet, sgd_optimizer, criterion, train_loader, device) 

                               
            for perm_level in PERM_LEVEL: 
                print(f"Permutation level: {perm_level}")

                train_loader, _ = data.dataload(batch_size=BATCH_SIZE, corrupt_prob=corrupt_prob, perm_level=perm_level)
                criterion = nn.CrossEntropyLoss()

                for opt in OPTIMIZER: 
                    if opt == "rmsprop": 
                        print(f"Model: RMS")
                        alexnet, device = model.loadmodel()
                        rms_optimizer = optim.RMSprop(alexnet.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
                        rms_accuracies, rms_losses = train.train(alexnet, rms_optimizer, criterion, train_loader, device)
                    if opt == "adam":
                        print("Model: Adam")
                        alexnet, device = model.loadmodel()
                        adam_optimizer = optim.Adam(alexnet.parameters(), lr=LEARNING_RATE_ADAM)
                        _,_ = train.train(alexnet, adam_optimizer, criterion, train_loader, device) 
                    if opt == "sgd": 
                        print("Model: SGD")
                        alexnet, device = model.loadmodel()
                        sgd_optimizer = optim.SGD(alexnet.parameters(), lr=LEARNING_RATE)
                        _,_ = train.train(alexnet, sgd_optimizer, criterion, train_loader, device) 
        
        else: 
            train_loader, _ = data.dataload(batch_size=BATCH_SIZE, corrupt_prob=corrupt_prob)
            alexnet, device = model.loadmodel()
            criterion = nn.CrossEntropyLoss()

            for opt in OPTIMIZER: 
                if opt == "rmsprop": 
                    print(f"Model: RMS")
                    alexnet, device = model.loadmodel()
                    rms_optimizer = optim.RMSprop(alexnet.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
                    rms_accuracies, rms_losses = train.train(alexnet, rms_optimizer, criterion, train_loader, device)
                if opt == "adam":
                    print("Model: Adam")
                    alexnet, device = model.loadmodel()
                    adam_optimizer = optim.Adam(alexnet.parameters(), lr=LEARNING_RATE_ADAM)
                    _,_ = train.train(alexnet, adam_optimizer, criterion, train_loader, device) 
                if opt == "sgd": 
                    print("Model: SGD")
                    alexnet, device = model.loadmodel()
                    sgd_optimizer = optim.SGD(alexnet.parameters(), lr=LEARNING_RATE)
                    _,_ = train.train(alexnet, sgd_optimizer, criterion, train_loader, device) 


if __name__ == "__main__": 
    main()

