# hyperspectral-autoencoders
Tools for training and using unsupervised autoencoders for hyperspectral data. 

Autoencoders are unsupervised neural networks that are useful for a range of applications such as unsupervised feature learning and dimensionality reduction. This repository provides a python-based toolbox with examples for building, training and testing both dense and convolutional autoencoders, designed for hyperspectral data. Networks are easy to setup and can be customised with different architectures. Different methods of training can also be implemented. It is built on tensorflow. 

![Alt text](images/diagram.png?raw=true "Hyperspectral Autoencoder")

If you use the cosine spectral angle (CSA) loss function in your research, please cite: 
[Windrim et al. **Unsupervised feature learning for illumination robustness.** 2016 IEEE International Conference on Image Processing (ICIP).](https://ieeexplore.ieee.org/abstract/document/7533202)

If you use the spectral angle (SA) or spectral information divergence (SID) loss function in your research, please cite:
[Windrim et al. **Unsupervised Feature-Learning for Hyperspectral Data with Autoencoders.** Remote Sensing 11.7 (2019): 864.](https://www.mdpi.com/2072-4292/11/7/864)


## Prerequisites

The software dependencies needed to run the toolbox are python 2.7 (tested with version 2.7.15) with packages:
* tensorflow (working with v1.14.0)
* numpy (working with v1.15.4)

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

A class within the toolbox from the data module called HypImg handles the dataset. The class accepts the hyperspectral data  in numy format, with shape [numRows x numCols x numBands] or [numSamples x numBands]. 

```
import data
hypData = data.HypImg( img )
```
Then the data can be pre-processed using a function of the HypImg class. For example, using the 'minmax' method:
```
hypData.pre_process( 'minmax' )
```
The result is stored in the attribute:
```
hypData.spectraPrep
```

Some hyperspectral datasets in a matlab file format (.mat) can be downloaded from [here](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes). A matlab file (.mat) can be converted to the numpy format using the [scipy.io.loadmat](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html) function.

### Data iterator

The Iterator class within the data module has methods for calling batches from the data that are used to train the network. A separate iterator object is made for the training and validation data. For example, an iterator object made from 100 pre-processed hyperspectral training samples with a batchsize of 10 is defined as:
```
dataTrain = data.Iterator( dataSamples=hypData.spectraPrep[:100, :],
targets=hypData.spectraPrep[:100, :], batchSize=10 )
```
For a typical autoencoder, the targets that the network is learning to output are the same as the data samples being input into the network. Similarly, an iterator object made from 20 validation samples is defined as:
```
dataVal = data.Iterator( dataSamples=hypData.spectraPrep[100:120, :],
targets=hypData.spectraPrep[100:120, :] )
```
Because the batchsize is unspecified, all 20 samples are used for the batch. The data in an iterator can also be shuffled before it is used to train a network:
```
dataTrain.shuffle()
```

### Building networks

The autoencoder module has classes used for creating autoencoder neural networks. There are currently two type of autoencoders that can be set up. An MLP autoencoder has purely fully-connected  (i.e. dense) layers:
```
net = autoencoder.mlp_1D_network( inputSize=hypData.numBands )
```
And a convolutional autoencoder has mostly convolutional layers, with a fully-connected layer used to map the final convolutional layer in the encoder to the latent vector:
```
net = autoencoder.cnn_1D_network( inputSize=hypData.numBands )
```
If not using config files to set up a network, then the input size of the data must be specified. This should be the number of spectral bands. 

Additional aspects of the network architecture can also be specified when initialising the object. For the MLP autoencoder:
```
net = autoencoder.mlp_1D_network( inputSize=hypData.numBands, encoderSize=[50,30,10,5],
activationFunc='relu', weightInitOpt='truncated_normal', tiedWeights=[1,0,0,0],
skipConnect=False, activationFuncFinal='linear')
```
- number of layers in the encoder (and decoder) - this is the length of the list 'encoderSize'
- number of neurons in each layer of the encoder - these are the values in the 'encoderSize' list. The last value in the list is the number of dimensions in the latent vector.
- the activation function which proceeds each layer and the function for the final decoder layer - activationFunc and activationFuncFinal
- the method of initialising network parameters - weightInitOpt
- which layers of the encoder to tie  to the decoder, such that they share a set of parameters - these are the values in the list 'tiedWeights'
- whether the network uses skip connections between corresponding layers in the encoder and decoder - specified by the boolean argument skipConnect


The convolutional autoencoder has similar arguments for defining the network architecture, but without 'encoderSize' and with some additional arguments:
```
net = autoencoder.cnn_1D_network( inputSize=hypData.numBands, zDim=3,
encoderNumFilters=[10,10,10], encoderFilterSize=[20,10,10], 
activationFunc='relu', weightInitOpt='truncated_normal', 
encoderStride=[1, 1, 1], padding='VALID', tiedWeights=[1,0,0,], 
skipConnect=False, activationFuncFinal='linear' )
```
- number of layers in the encoder (and decoder) - this is the length of the list 'encodernumFilters'
- number of filters/kernels in each conv layer - these are the values in the 'encodernumFilters' list
- the size of the filters/kernels in each conv layer - these are the values in the 'encoderFilterSize' list
- the stride of the filters/kernels in each conv layer - these are the values in the 'encoderStride' list
- the number of dimensions in the latent vector - zDim
- the type of padding each conv layer uses - padding


Alternatively to defining the architecture by the initialisation arguments, a config.json file can be used:
```
mlp = autoencoder.mlp_1D_network( configFile='config.json') )
```
Some example config files can be found in examples/example_configs.













