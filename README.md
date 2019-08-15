# hyperspectral-autoencoders
Tools for training and using unsupervised autoencoders for hyperspectral data. 

Autoencoders are unsupervised neural networks that are useful for a range of applications such as unsupervised feature learning and dimensionality reduction. This repository provides a python-based toolbox with examples for building, training and testing both dense and convolutional autoencoders, designed for hyperspectral data. Networks are easy to setup and can be customised with different architectures. Different methods of training can also be implemented. It is built on tensorflow. 

If you use the cosine spectral angle (CSA) loss function in your research, please cite: 
[Windrim et al. **Unsupervised feature learning for illumination robustness.** 2016 IEEE International Conference on Image Processing (ICIP).](https://ieeexplore.ieee.org/abstract/document/7533202)

If you use the spectral angle (SA) or spectral information divergence (SID) loss function in your research, please cite:
[Windrim et al. **Unsupervised Feature-Learning for Hyperspectral Data with Autoencoders.** Remote Sensing 11.7 (2019): 864.](https://www.mdpi.com/2072-4292/11/7/864)


## Prerequisites

The software dependencies needed to run the toolbox are python 2.7 (tested with version 2.7.15) with packages:
* tensorflow (working with v1.14.0)
* numpy

Each of these packages can be installed using [pip](https://pypi.org/project/pip/). The example scripts use some additional packages such as scipy and matplotlib. 

## Quickstart
To start training an autoencoder right away, run the example script:
```
train_MLP_basic.py
```
It will download the Pavia Uni dataset and train an autoencoder. You can then run the example script:
```
_test_MLP_basic.py 
```
to test the autoencoder that was just trained, and generate some images of the latent hyperspectral image, latent vector and comparisons of the spectra and reconstruction in the results folder.

## Usage

The toolbox compresses several key processes:
- data preparation
- data iterator
- building networks
- adding train operations
- training networks

Each of these are elaborated on below:

### Data preparation

The toolbox uses a class from the data module called HypImg to prepare the dataset. 

The toolbox uses hyperspectral data in a numpy format. If you are using










