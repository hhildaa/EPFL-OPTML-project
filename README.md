# EPFL Course - Optimization for Machine Learning Mini Project - CS-439

## Abstract
The goal of the project was to compare different optimisers on images with corrupted labels and corrupted images to examine how much neural networks can memorise a big amount of data. We found that while the networks are able to fit and obtain 100\% train accuracy even with fully randomized labels, the selection of optimiser can have a significant effect on the training time. However, the generalisation error is not effected by the choice of optimiser. This project was a part of the Optimization for Machine Learning course at EPFL in 2022. 

## Applied corruptions:
- **True labels:** Original dataset with no corruptions.
- **Partially corrupted labels:** Each label is corrupted with probability p, where p ∈ [0.2, 0.4, 0.6, 0.8, 1]
- **Shuffled pixels:** The same pixel permutation is applied to the entire dataset. This is done in 2 levels, the permutation is applied along a single axis or along both axis.
- **Random noise:** A Gaussian distribution with same mean and variance as the original dataset is used to generate a new dataset.

## Files
- [**data.py**](data.py): Loads randomly selected 25.000 train images from CIFAR10 dataset by applying the selected corruption.
- [**util.py**](util.py): Includes utility functions to 
  - Load pretrained AlexNet modified for 10 output classes,
  - Train the model with selected optimiser until convergence to 100% accuracy.
- [**test.py**](test.py): Tests the performance of trained model with uncorrupted test set.

- [**SGD.ipynb**](SGD.ipynb): Experiment results for SGD optimiser.
- [**ADAM.ipynb**](ADAM.ipynb): Experiment results for ADAM optimiser.
- [**RMSprop.ipynb**](RMSprop.ipynb): Experiment results for RMSprop optimiser.

## Dependency


## Reproductibility
To reproduce our experiment results, one can run the run.py file with the following parameters.
```
  CORRUPT_PROB = [0, 0.2, 0.4, 0.6, 0.8, 1]
  NOISE = [True, False]
  PERM_LEVEL = [0, 1, 2]
  OPTIMIZER = ["adam", "sgd", "rmsprop"]            # possible values: "adam", "sgd", "rmsprop"
```
As running the experiments computational heavy, we used GPU for experiments for ADAM and Google collab notebooks for SGD and RMSprop.
These notebooks are also available in the repository under `SGD.ipynb` and `RMSprop.ipynb` names.

For reproducing our plots, the parameters should be chosen the following. 
Plot for corrupted labels:
```
CORRUPT_PROB = [0, 0.2, 0.4, 0.6, 0.8, 1]
NOISE = [False]
PERM_LEVEL = []
```
Plot for corrupted images: 
```
CORRUPT_PROB = [0, 1]
NOISE = [False, True]
PERM_LEVEL = [1, 2]
```

## Authors:
- Hilda Abigél Horváth
- Nicholas Sperry Grandhomme
- Medya Tekeş Mızraklı
