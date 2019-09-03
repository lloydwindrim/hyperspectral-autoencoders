# deephyp: deep learning for hyperspectral 

![PyPI](https://img.shields.io/pypi/v/deephyp) ![PyPI - Downloads](https://img.shields.io/pypi/dm/deephyp?color=brightgreen)

Tools for training and using unsupervised autoencoders and supervised deep learning classifiers for hyperspectral data.

Documentation available [here](https://deephyp.readthedocs.io/en/latest/index.html). Source code available on [Github](https://github.com/lloydwindrim/hyperspectral-autoencoders).

Autoencoders are unsupervised neural networks that are useful for a range of applications such as unsupervised feature learning and dimensionality reduction. Supervised deep learning classifiers can be trained on labelled data to predict the class of spectra. This repository provides a python-based toolbox called **deephyp**, with examples for building, training and testing both dense and convolutional autoencoders and classification neural networks, designed for hyperspectral data. Networks are easy to setup and can be customised with different architectures. Different methods of training can also be implemented. The toolbox is built on tensorflow.

![Alt text](images/diagram.png?raw=true "Hyperspectral Autoencoder")

If you use the toolbox in your research, please cite:
[Windrim et al. **Unsupervised Feature-Learning for Hyperspectral Data with Autoencoders.** Remote Sensing 11.7 (2019): 864.](https://www.mdpi.com/2072-4292/11/7/864)
This paper explains the spectral angle (SA), spectral information divergence (SID) and sum-of-squared errors (SSE) loss functions for training autoencoders.

If you use the cosine spectral angle (CSA) loss function in your research, please cite: 
[Windrim et al. **Unsupervised feature learning for illumination robustness.** 2016 IEEE International Conference on Image Processing (ICIP).](https://ieeexplore.ieee.org/abstract/document/7533202)

If you use the classification networks in your research, please cite:
[Windrim et al. **Hyperspectral CNN Classification with Limited Training Samples.** 2017 Proceedings of the British Machine Vision Conference (BMVC).](https://www.researchgate.net/publication/332818169_Hyperspectral_CNN_Classification_with_Limited_Training_Samples)

## Installation

The [latest release](https://pypi.org/project/deephyp/) of the toolbox can be installed from the command line using pip:
```
pip install deephyp
```
Dependencies can be installed using:
```
pip install -r requirements.txt
```
To import:
```
import deephyp
```

## Prerequisites

The software dependencies needed to run the toolbox are python 2 or python 3 (tested with version 2.7.15 and 3.5.2) with packages:
* tensorflow (tested with v1.14.0) 
* numpy (tested with v1.15.4)

Each of these packages can be installed using [pip](https://pypi.org/project/pip/). The example scripts use some additional packages such as scipy (tested with v1.3.1) and matplotlib (tested with v3.0.3).

## Quickstart
To start training an autoencoder right away, move to the /examples directory in the [github repo](https://github.com/lloydwindrim/hyperspectral-autoencoders) and run the example script:
```
autoencoder_train_MLP_basic.py
```
It will download the Pavia Uni dataset and train an autoencoder. You can then run the example script:
```
autoencoder_test_MLP_basic.py
```
to test the autoencoder that was just trained, and generate some images of the latent hyperspectral image, latent vector and comparisons of the spectra and reconstruction in the results folder.

To start training a classification network, from the /examples directory run the script:
```
classifier_train_CNN_basic.py
```
which will download the Pavia Uni dataset and ground truth labels and train a classifier. Then running the example script:
```
classifier_test_CNN_basic.py
```
will load the trained classifier and generate images with the predicted and ground truth labels for comparison.

## Usage

The toolbox comprises several key processes:
- [data preparation](#data-preparation)
- [data iterator](#data-iterator)
- [building networks](#building-networks)
- [adding train operations](#adding-training-operations)
- [training networks](#training-networks)
- [loading a trained network](#loading-a-trained-network)

Each of these are elaborated on below:

### Data preparation

A class within the toolbox from the data module called HypImg handles the dataset. The class accepts the hyperspectral data  in numpy format, with shape [numRows x numCols x numBands] or [numSamples x numBands]. The networks in the toolbox operate in the spectral domain, not the spatial, so if an image is input with shape [numRows x numCols x numBands], it is reshaped to [numSamples x numBands], collapsing the spatial dimensions into one.

```
from deephyp import data
hypData = data.HypImg( img )
```
or for hyperspectral data with class labels (i.e. for supervised classification):
```
hypData = data.HypImg( img, labels )
```
Then the data can be pre-processed using a function of the HypImg class. For example, using the 'minmax' approach:
```
hypData.pre_process( 'minmax' )
```
The result is stored in the attribute:
```
hypData.spectraPrep
```
When provided, labels are converted to a one-hot format stored as:
```
hypData.labelsOnehot
```
Any labels with value <= 0 are considered as a background class, and appear as a row of zeros in labelsOnehot. They also do not count towards the number of classes stored in the data handler.

Some hyperspectral datasets in a matlab file format (.mat) can be downloaded from [here](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes). A matlab file (.mat) can be converted to the numpy format using the [scipy.io.loadmat](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html) function.

### Data iterator

The Iterator class within the data module has methods for calling batches from the data that are used to train the network. A separate iterator object is made for the training and validation data. For example, an iterator object made from 100 pre-processed hyperspectral training samples with a batchsize of 10 is defined as:
```
dataTrain = data.Iterator( dataSamples=hypData.spectraPrep[:100, :], targets=hypData.spectraPrep[:100, :], batchSize=10 )
```
Similarly, an iterator object made from 20 validation samples is defined as:
```
dataVal = data.Iterator( dataSamples=hypData.spectraPrep[100:120, :], targets=hypData.spectraPrep[100:120, :] )
```
Because the batchsize is unspecified, all 20 samples are used for the batch.

For a typical unsupervised autoencoder, the targets that the network is learning to output are the same as the data samples being input into the network, as in the above iterator examples. When training a supervised classifier, the targets are the one-hot class labels:
```
dataTrain = data.Iterator( dataSamples=hypData.spectraPrep[:100, :], targets=hypData.labelsOnehot[:100, :], batchSize=10 )
```
The data in any iterator can also be shuffled before it is used to train a network:
```
dataTrain.shuffle()
```

### Building networks

The autoencoder module has classes used for creating autoencoder neural networks:
```
from deephyp import autoencoder
```
There are currently two type of autoencoders that can be set up. An MLP autoencoder has purely fully-connected  (i.e. dense) layers:
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
net = autoencoder.mlp_1D_network( inputSize=hypData.numBands, encoderSize=[50,30,10,5], activationFunc='relu', weightInitOpt='truncated_normal', tiedWeights=[1,0,0,0], skipConnect=False, activationFuncFinal='linear')
```
- number of layers in the encoder (and decoder) - this is the length of the list 'encoderSize'
- number of neurons in each layer of the encoder - these are the values in the 'encoderSize' list. The last value in the list is the number of dimensions in the latent vector.
- the activation function which proceeds each layer and the function for the final decoder layer - activationFunc and activationFuncFinal
- the method of initialising network parameters (e.g. xavier improved) - 'weightInitOpt'
- which layers of the encoder to tie  to the decoder, such that they share a set of parameters - these are the values in the list 'tiedWeights'
- whether the network uses skip connections between corresponding layers in the encoder and decoder - specified by the boolean argument skipConnect


The convolutional autoencoder has similar arguments for defining the network architecture, but without 'encoderSize' and with some additional arguments:
```
net = autoencoder.cnn_1D_network( inputSize=hypData.numBands, zDim=3, encoderNumFilters=[10,10,10], encoderFilterSize=[20,10,10],  activationFunc='relu', weightInitOpt='truncated_normal',  encoderStride=[1, 1, 1], padding='VALID', tiedWeights=[1,0,0],  skipConnect=False, activationFuncFinal='linear' )
```
- number of layers in the encoder (and decoder) - this is the length of the list 'encodernumFilters'
- number of filters/kernels in each conv layer - these are the values in the 'encodernumFilters' list
- the size of the filters/kernels in each conv layer - these are the values in the 'encoderFilterSize' list
- the stride of the filters/kernels in each conv layer - these are the values in the 'encoderStride' list
- the number of dimensions in the latent vector - zDim
- the type of padding each conv layer uses - padding


Alternatively to defining the architecture by the initialisation arguments, a config.json file can be used:
```
net = autoencoder.mlp_1D_network( configFile='config.json') )
```
Some example config files can be found in examples/example_configs.

The classifier module has classes used for creating supervised classification neural networks:
```
from deephyp import classifier
```
There is currently one type of classifier that can be set up, which contains a combination of convolutional layers (at the start) and fully-connected layers (at the end):
```
net = classifier.cnn_1D_network( inputSize=hypData.numBands, numClasses=hypData.numClasses )
```
If not using config files the input size and number of classes must be specified for classifier networks. As with the autoencoder class, additional aspects of the architecture can be specified:
```
net = classifier.cnn_1D_network( inputSize=hypData.numBands, numClasses=hypData.numClasses, convFilterSize=[20,10,10], convNumFilters=[10,10,10], convStride = [1,1,1], fcSize=[20,20], activationFunc='relu', weightInitOpt='truncated_normal', padding='VALID' )
```
- number of convolutional layers - this is the length of the list 'convNumFilters'
- number of filters/kernels in each conv layer - these are the values in the 'convNumFilters' list
- the size of the filters/kernels in each conv layer - these are the values in the 'convFilterSize' list
- the stride of the filters/kernels in each conv layer - these are the values in the 'convStride' list
- the type of padding each conv layer uses - padding
- number of fully-connected layers - this is the length of the list 'fcSize'
- number of neurons in each fully-connected layer - these are the values in the 'fcSize' list
- the activation function which proceeds each layer - activationFunc
- the method of initialising network parameters (e.g. xavier improved) - 'weightInitOpt'

Classifiers can also be setup with config files.

### Adding training operations

Once a network has been created, a training operation can be added to it. It is possible to add multiple training operations to a network, so each op must be given a name:
```
net.add_train_op( name='experiment_1' )
```
When adding a train op, details about how the network will be trained with that op can be specified. For example, a train op for an autoencoder which uses the cosine spectral angle (CSA) loss function, a learning rate of 0.001 with no decay, optimised with Adam and no weight decay can be defined by:
```
net.add_train_op( name='experiment_1', lossFunc='CSA', learning_rate=1e-3, decay_steps=None, decay_rate=None, method='Adam', wd_lambda=0.0 )
```
There are several loss functions that can be used to train an autoencoder, many of which were designed specifically for hyperspectral data:
- [cosine spectral angle (CSA)](https://ieeexplore.ieee.org/abstract/document/7533202)
- [spectral angle (SA)](https://www.mdpi.com/2072-4292/11/7/864)
- [spectral information divergence (SID)](https://www.mdpi.com/2072-4292/11/7/864)
- [sum-of-squared errors (SSE)](https://www.mdpi.com/2072-4292/11/7/864)

Note that when using the CSA, SA and SID loss functions it is expected that the reconstructed spectra have a different magnitude to the target spectra, but a similar shape. The SSE should produce a similar magnitude and shape. Also, since the SID contains *log* in its expression which is undefined for values *<= 0*, it is best to use sigmoid as the activation function (including the final activation function) for networks trained with the SID loss. See train_MLP_sid.py for an example.

The method of decaying the learning rate can also be customised. For example, to decay exponentially every 100 steps:
```
net.add_train_op( name='experiment_1',learning_rate=1e-3, decay_steps=100, decay_rate=0.9 )
```
A piecewise approach of decaying the learning rate can also be used. For example, to change the learning rate from 0.001 to 0.0001 after 100 steps, and then to 0.00001 after a further 200 steps:
```
net.add_train_op( name='experiment_1',learning_rate=1e-3, piecewise_bounds=[100,300], piecewise_values=[1e-4,1e-5] )
```
Train ops are added to classifiers in the same way, with the only difference being the option to balance classes when training, and no specification of the loss function (a cross-entropy loss function is used):
```
net.add_train_op( name='basic_clf', balance_classes=True, learning_rate=1e-3 )
```

### Training networks

Once one or multiple training ops have been added to a network, they can be used to learn a model (or multiple models) for that network through training:
```
net.train(dataTrain=dataTrain, dataVal=dataVal, train_op_name='experiment_1', n_epochs=100, save_addr=model_directory, visualiseRateTrain=5, visualiseRateVal=10, save_epochs=[50,100])
```
The train method learns a model using one train op, therefore the train method should be called at least once for each train op that was added. The name of the train op must be specified, and the training and validation iterators created previously must be input. A path to a directory to save the model must also be specified. The example above will train a network for 100 epochs of the training dataset, and save the model at 50 and 100 epochs. The training loss will be displayed every 5 epochs, and the validation loss will be displayed every 10 epochs.

It is also possible to load a pre-trained model and continue to train it by passing the address of epoch folder containing the model checkpoint as the save_addr argument. For example, if the directory for the model at epoch 50 (epoch_50 folder) was passed to save_addr in the example above, then the model would be trained for an additional 50 epochs to reach 100, and it would be saved in a folder called epoch_100 in the same directory as the epoch_50 folder.

The interface for training autoencoders and classifiers is the same.

### Loading a trained network

To load a trained model on a new dataset, ensure the data has been pre-processed similarly using:

```
import data
hypData = data.HypImg( new_img )
hypData.pre_process( 'minmax' )
```
Then set up the network. The network architecture must be the same as the one used for the model being loaded. However, this is easy as the directory where models are saved should contain an automatically generated config.json file, which can be used to set up the network with the same architecture:
```
net = autoencoder.mlp_1D_network( configFile='model_directory/config.json' )
```
Once the architecture has been defined, add a model to the network:
```
net.add_model( addr='model_directory/epoch_100'), modelName='csa_100' )
```
Because multiple models can be added to a single network, the added model must be given a name.

When the network is set up and a model has been added, hyperspectral data can be passed through it. To use a trained autoencoder to extract the latent vectors of some spectra:
```
dataZ = net.encoder( modelName='csa_100', dataSamples=hypData.spectraPrep )
```
Make sure to refer to the name of the model the network should use. The encoded hyperspectral (dataZ) data can also be decoded to get the reconstruction:
```
dataY = net.decoder(modelName='csa_100', dataZ=dataZ)
```
It is also possible to encode and decode in one step with:
```
dataY = net.encoder_decoder(modelName='csa_100', dataZ=hypData.spectraPrep)
```
To use a trained classifier to predict the classification labels of some spectra:
```
dataPred = net.predict_labels( modelName='basic_model', dataSamples=hypData.spectraPrep  )
```
To extract the features in the second last layer of the classifier network:
```
dataFeatures = net.predict_features(modelName='basic_model', dataSamples=hypData.spectraPrep, layer=net.numLayers-1)
```

## Results

An example of a latent space for the Pavia University dataset, produced with a MLP autoencoder trained using the cosine spectral angle (CSA):

![Alt text](images/mlp_latent_space.png?raw=true "autoencoder MLP latent space")

And an example of a latent space for the Pavia University dataset, produced with a convolutional autoencoder trained using the cosine spectral angle (CSA):

![Alt text](images/cnn_latent_space.png?raw=true "autoencoder CNN latent space")

Both figures were made running the scripts:
```
autoencoder_train_CNN_vs_MLP.py
autoencoder_test_CNN_vs_MLP.py
```

A classification result for the Pavia University dataset produced using the CNN classifier:

![Alt text](images/classification_pred.png?raw=true "CNN classification result")

The figure was made by running the scripts:
```
classifier_train_CNN_basic.py
classifier_test_CNN_basic.py
```


## Related publications

Some links to publications on deep learning for hyperspectral data:

- autoencoders: [ICIP 2016](https://ieeexplore.ieee.org/abstract/document/7533202), [TIP 2017](https://ieeexplore.ieee.org/abstract/document/8063434), [Remote Sensing 2019](https://www.mdpi.com/2072-4292/11/7/864)
- CNNs for classification using data augmentation: [BMVC 2017](https://www.researchgate.net/publication/332818169_Hyperspectral_CNN_Classification_with_Limited_Training_Samples)
- pre-training CNNs: [TGRS 2018](https://ieeexplore.ieee.org/abstract/document/8245897)
- [PhD thesis](https://ses.library.usyd.edu.au/handle/2123/18734)
