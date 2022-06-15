# EPFL Course - Optimization for Machine Learning Mini Project - CS-439
This project compares the performance of different optimisers on images with corrupted labels and corrupted images through a series of systematic experiments.

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

## Authors:
- Hilda Abigél Horváth
- Nicholas Sperry Grandhomme
- Medya Tekeş Mızraklı
