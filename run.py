# imports
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
import plot

# PARAMETERS
LEARNING_RATE = 0.001
LEARNING_RATE_ADAM = 0.00001
LEARNING_RATE_RMS = 0.00001
BATCH_SIZE = 64

# RUNNING PARAMETERS
CORRUPT_PROB = [0]
NOISE = [True, False]
PERM_LEVEL = [0, 1, 2]
OPTIMIZER = ["rmsprop"]            # possible values: "adam", "sgd", "rmsprop"

# set random seed for REPRODUCIBILITY
SEED = 2022
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def main():
    """
    Run experiments. 
    """
    # save results for plots
    label_accs = {opt: [] for opt in OPTIMIZER}
    image_accs = {opt: [] for opt in OPTIMIZER}
    gen_errors = {opt: [] for opt in OPTIMIZER}
    
    # run experiments for different label corruption probabilities
    for corrupt_prob in CORRUPT_PROB: 
        print ("------------------------")
        print(f"Corruption probability: {corrupt_prob}")

        # run experiments without label corruption
        if corrupt_prob == 0: 
            
            # run experiments with or without Gaussian noise
            for noise in NOISE: 
                print(f"Noise: {noise}")

                # load data
                train_loader, test_loader = data.dataload(batch_size=BATCH_SIZE, corrupt_prob=corrupt_prob, perm_level=0, random_noise=noise)
                criterion = nn.CrossEntropyLoss()

                # run for the different optimizers
                for opt in OPTIMIZER: 
                    if opt == "rmsprop": 
                        print(f"Model: RMS")
                        alexnet, device = model.loadmodel()
                        rms_optimizer = optim.RMSprop(alexnet.parameters(), lr=LEARNING_RATE_RMS, weight_decay=1e-4)
                        accs, _, optmodel = train.train(alexnet, rms_optimizer, criterion, train_loader, device)
                        error = test.test(optmodel, test_loader, device)
                        
                    if opt == "adam":
                        print("Model: Adam")
                        alexnet, device = model.loadmodel()
                        adam_optimizer = optim.Adam(alexnet.parameters(), lr=LEARNING_RATE_ADAM)
                        accs, _, optmodel = train.train(alexnet, adam_optimizer, criterion, train_loader, device) 
                        error = test.test(optmodel, test_loader, device)

                    if opt == "sgd": 
                        print("Model: SGD")
                        alexnet, device = model.loadmodel()
                        sgd_optimizer = optim.SGD(alexnet.parameters(), lr=LEARNING_RATE, momentum = 0.9, weight_decay=1e-4)
                        accs, _, optmodel = train.train(alexnet, sgd_optimizer, criterion, train_loader, device) 
                        error = test.test(optmodel, test_loader, device)

                    # save results for plot
                    if noise:
                        image_accs[opt].append(accs)
                    else:
                        label_accs[opt].append(accs)
                        #We also want ground truth when plotting the image corruptions
                        image_accs[opt].append(accs)
                        gen_errors[opt].append(100-error)

            # run experiments for pixel permutations               
            for perm_level in PERM_LEVEL: 
                print(f"Permutation level: {perm_level}")

                # load data
                train_loader, test_loader = data.dataload(batch_size=BATCH_SIZE, corrupt_prob=corrupt_prob, perm_level=perm_level,)
                criterion = nn.CrossEntropyLoss()

                # run for the different optimizers
                for opt in OPTIMIZER: 
                    if opt == "rmsprop": 
                        print(f"Model: RMS")
                        alexnet, device = model.loadmodel()
                        rms_optimizer = optim.RMSprop(alexnet.parameters(), lr=LEARNING_RATE_RMS, weight_decay=1e-4)
                        accs, _, optmodel = train.train(alexnet, rms_optimizer, criterion, train_loader, device)
                        error = test.test(optmodel, test_loader, device)
                    if opt == "adam":
                        print("Model: Adam")
                        alexnet, device = model.loadmodel()
                        adam_optimizer = optim.Adam(alexnet.parameters(), lr=LEARNING_RATE_ADAM)
                        accs, _, optmodel = train.train(alexnet, adam_optimizer, criterion, train_loader, device) 
                        error = test.test(optmodel, test_loader, device)
                    if opt == "sgd": 
                        print("Model: SGD")
                        alexnet, device = model.loadmodel()
                        sgd_optimizer = optim.SGD(alexnet.parameters(), lr=LEARNING_RATE, momentum = 0.9, weight_decay=1e-4)
                        accs, _, optmodel = train.train(alexnet, sgd_optimizer, criterion, train_loader, device)
                        error = test.test(optmodel, test_loader, device)
                    
                    # save results for plot
                    image_accs[opt].append(accs)
        
        # run label corrupted experiments
        else: 

            # load data
            train_loader, test_loader = data.dataload(batch_size=BATCH_SIZE, corrupt_prob=corrupt_prob)
            alexnet, device = model.loadmodel()
            criterion = nn.CrossEntropyLoss()

            # run for the different optimizers
            for opt in OPTIMIZER: 
                if opt == "rmsprop": 
                    print(f"Model: RMS")
                    alexnet, device = model.loadmodel()
                    rms_optimizer = optim.RMSprop(alexnet.parameters(), lr=LEARNING_RATE_RMS, weight_decay=1e-4)
                    accs, _, optmodel = train.train(alexnet, rms_optimizer, criterion, train_loader, device)
                    error = test.test(optmodel, test_loader, device)
                    
                if opt == "adam":
                    print("Model: Adam")
                    alexnet, device = model.loadmodel()
                    adam_optimizer = optim.Adam(alexnet.parameters(), lr=LEARNING_RATE_ADAM)
                    accs, _, optmodel = train.train(alexnet, adam_optimizer, criterion, train_loader, device) 
                    error = test.test(optmodel, test_loader, device)
                    
                if opt == "sgd": 
                    print("Model: SGD")
                    alexnet, device = model.loadmodel()
                    sgd_optimizer = optim.SGD(alexnet.parameters(), lr=LEARNING_RATE, momentum = 0.9, weight_decay=1e-4)
                    accs, _, optmodel = train.train(alexnet, sgd_optimizer, criterion, train_loader, device)
                    error = test.test(optmodel, test_loader, device)
                
                # save results for plot
                label_accs[opt].append(accs)
                gen_errors[opt].append(100-error)
                if corrupt_prob == 1:
                    #We also want 100% random labels when plotting the image corruptions
                    image_accs[opt].append(accs)
    plot.save_plots(label_accs, image_accs, gen_errors)


if __name__ == "__main__": 
    main()

