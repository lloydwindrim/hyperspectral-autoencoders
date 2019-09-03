.. deephyp documentation master file, created by
   sphinx-quickstart on Thu Aug 29 19:50:37 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Getting started with autoencoders
=================================

Autoencoders are unsupervised neural networks that are useful for a range of applications such as unsupervised feature learning and dimensionality reduction. Autoencoders are trained to learn the parameters for an *encoder* which maps the input data to a latent space and a *decoder* which reconstructs the input from the latent space. The latent space is often of a lower dimensionality then the input data, and can be thought of as a feature vector. The network is trained to minimise the reconstruction error between its decoded output and the input data (hence it is *unsupervised*). Once trained, the *encoder* can be used to map data to the latent space.

.. image:: cnn_latent_space.png

Download a hyperspectral dataset
--------------------------------

Some hyperspectral datasets in a matlab file format (.mat) can be downloaded from `here
<http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes>`_. To get started, download the 'Pavia University' dataset.

**deephyp** operates on hyperspectral data in numpy array format. The matlab file (.mat) you just downloaded can be read as a numpy array using the `scipy.io.loadmat
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html>`_ function:

.. code-block:: python

    import scipy.io
    mat = scipy.io.loadmat( 'PaviaU.mat' )
    img = mat[ 'paviaU' ]

where *img* is a numpy array. You are now ready to use the toolbox!

Overview
--------

For both autoencoders and classifiers, the toolbox uses several key processes:

- data preparation
- data iterator
- building networks
- adding train operations
- training networks
- loading and testing a trained network

Each of these are elaborated on below:

Data preparation
----------------

A class within the toolbox from the *data* module called *HypImg* handles the hyperspectral dataset and all of its meta-data. As mentioned earlier, the class accepts the hyperspectral data in numpy format, with shape [numRows x numCols x numBands] or [numSamples x numBands]. The networks in the toolbox operate in the spectral domain, not the spatial, so if a hypercube image is input with shape [numRows x numCols x numBands], it is reshaped to [numSamples x numBands], collapsing the spatial dimensions into a single dimension.

The Pavia Uni hyperspectral image can be passed to the *HypImg* class as follows:

.. code-block:: python

   from deephyp import data
   hypData = data.HypImg( img )

It is also possible to pass class labels to *HypImg*, but if you are training an unsupervised autoencoder you do not need to do this.

Then the data can be pre-processed using a function of the *HypImg* class. For example, using the 'minmax' approach:

.. code-block:: python

   hypData.pre_process( 'minmax' )

The result is stored in the attribute *spectraPrep* attribute. Currently, only the 'minmax' approach is available, but additions will be made in future versions.


Data iterator
-------------

The *Iterator* class within the *data* module has methods for calling batches from the data that are used to train the network. A separate iterator object is made for the training and validation data. For example, an iterator object made from 200,000 pre-processed hyperspectral training samples with a batchsize of 1000 is defined by:

.. code-block:: python

   dataTrain = data.Iterator( dataSamples=hypData.spectraPrep[:200000, :], targets=hypData.spectraPrep[:200000, :], batchSize=1000 )

Similarly, an iterator object made from 100 validation samples is defined as:

.. code-block:: python

   dataVal = data.Iterator( dataSamples=hypData.spectraPrep[200000:200100, :], targets=hypData.spectraPrep[200000:200100, :] )

Because the batchsize is unspecified for the validation iterator, all 100 samples are used for each batch. For a typical unsupervised autoencoder, the targets that the network is learning to output are the same as the data samples being input into the network, as in the above iterator examples. When training a supervised classifier, the targets will be the ground truth class labels.

The data in any iterator can also be shuffled before it is used to train a network:

.. code-block:: python

   dataTrain.shuffle()


Building networks
-----------------

The *autoencoder* module has classes for creating autoencoder neural networks:

.. code-block:: python

   from deephyp import autoencoder

There are currently two type of autoencoders that can be set up. A multi-layer perceptron (MLP) autoencoder has purely fully-connected (i.e. dense) layers:

.. code-block:: python

   net = autoencoder.mlp_1D_network( inputSize=hypData.numBands )

And a convolutional autoencoder has mostly convolutional layers, with a fully-connected layer used to map the final convolutional layer in the encoder to the latent vector:

.. code-block:: python

   net = autoencoder.cnn_1D_network( inputSize=hypData.numBands )

If not using config files to set up a network, then the input size of the data must be specified. This should be the number of spectral bands, which is stored in *hypData.numBands* for convenience.

Additional aspects of the network architecture can also be specified when initialising the *autoencoder* object. For the MLP autoencoder:

.. code-block:: python

   net = autoencoder.mlp_1D_network( inputSize=hypData.numBands, encoderSize=[50,30,10,5], activationFunc='relu', weightInitOpt='truncated_normal', tiedWeights=[1,0,0,0], skipConnect=False, activationFuncFinal='linear')


where the following components of the architecture can be specified:


- number of layers in the encoder (and decoder) - this is the length of the list 'encoderSize'
- number of neurons in each layer of the encoder - these are the values in the 'encoderSize' list. The last value in the list is the number of dimensions in the latent vector.
- the activation function which proceeds each layer and the function for the final decoder layer - activationFunc and activationFuncFinal
- the method of initialising network parameters (e.g. xavier improved) - 'weightInitOpt'
- which layers of the encoder to tie  to the decoder, such that they share a set of parameters - these are the values in the list 'tiedWeights'
- whether the network uses skip connections between corresponding layers in the encoder and decoder - specified by the boolean argument skipConnect

Therefore, the above MLP autoencoder has four encoder layers (and four symmetric decoder layers), with five neurons in the latent layer. This network could be used to represent a hyperspectral image with five dimensions.

The convolutional autoencoder has similar arguments for defining the network architecture, but without 'encoderSize' and with some additional arguments:

.. code-block:: python

   net = autoencoder.cnn_1D_network( inputSize=hypData.numBands, zDim=3, encoderNumFilters=[10,10,10], encoderFilterSize=[20,10,10],  activationFunc='relu', weightInitOpt='truncated_normal',  encoderStride=[1, 1, 1], padding='VALID', tiedWeights=[0,0,0],  skipConnect=False, activationFuncFinal='linear' )


which are:

- number of layers in the encoder (and decoder) - this is the length of the list 'encodernumFilters'
- number of filters/kernels in each conv layer - these are the values in the 'encodernumFilters' list
- the size of the filters/kernels in each conv layer - these are the values in the 'encoderFilterSize' list
- the stride of the filters/kernels in each conv layer - these are the values in the 'encoderStride' list
- the number of dimensions in the latent vector - zDim
- the type of padding each conv layer uses - padding

Note that the convolutional autoencoder uses *deconvolutional* layers in the decoder, which can upsample the data from the latent layer to the output layer.


Instead of defining the network architecture by the initialisation arguments, a config.json file can be used:

.. code-block:: python

   net = autoencoder.mlp_1D_network( configFile='config.json') )

A config file is generated each time a network in the toolbox is trained, so you can use one from another network as a template for making a new one.


Adding training operations
--------------------------

Once a network has been created, a training operation can be added to it. It is possible to add multiple training operations to a network, so each op must be given a name:

.. code-block:: python

   net.add_train_op( name='experiment_1' )

When adding a train op, details about how the network will be trained with that op can also be specified. For example, a train op for an autoencoder which uses the cosine spectral angle (CSA) loss function, a learning rate of 0.001 with no decay, optimised with Adam and no weight decay can be defined by:

.. code-block:: python

   net.add_train_op( name='experiment_1', lossFunc='CSA', learning_rate=1e-3, method='Adam', wd_lambda=0.0 )


There are several loss functions that can be used to train an autoencoder with this toolbox, many of which were designed specifically for hyperspectral data:

- cosine spectral angle (CSA)
- spectral angle (SA)
- spectral information divergence (SID)
- sum-of-squared errors (SSE)

Note that when using the `CSA
<https://ieeexplore.ieee.org/abstract/document/7533202>`_, `SA
<https://www.mdpi.com/2072-4292/11/7/864>`_ and `SID
<https://www.mdpi.com/2072-4292/11/7/864>`_ loss functions it is expected that the reconstructed spectra have a different magnitude to the target spectra, but a similar shape. The `SSE
<https://www.mdpi.com/2072-4292/11/7/864>`_ should produce a similar magnitude and shape. Also, since the SID contains *log* in its expression which is undefined for values *<= 0*, it is best to use sigmoid as the activation function (including the final activation function) for networks trained with the SID loss. See the code examples for a demonstration.

The method for decaying the learning rate can also be customised. For example, to decay the learning rate exponentially every 100 steps (starting at 0.001):

.. code-block:: python

   net.add_train_op( name='experiment_1',learning_rate=1e-3, decay_steps=100, decay_rate=0.9 )


A piecewise approach to decaying the learning rate can also be used. For example, to change the learning rate from 0.001 to 0.0001 after 100 steps, and then to 0.00001 after a further 200 steps:

.. code-block:: python

   net.add_train_op( name='experiment_1',learning_rate=1e-3, piecewise_bounds=[100,300], piecewise_values=[1e-4,1e-5] )


Training networks
-----------------

Once one or multiple training ops have been added to a network, they can be used to learn a model (or multiple models) for that network through training:

.. code-block:: python

   net.train( dataTrain=dataTrain, dataVal=dataVal, train_op_name='experiment_1', n_epochs=100, save_addr=model_directory, visualiseRateTrain=5, visualiseRateVal=10, save_epochs=[50,100])

The train method learns a model using one train op, therefore the train method should be called at least once for each train op that was added. The name of the train op must be specified, and the training and validation iterators created previously must be input. A path to a directory to save the model must also be specified. The example above will train a network for 100 epochs of the training dataset (that is, loop through the entire training dataset 100 times), and save the model at 50 and 100 epochs. The training loss will be displayed every 5 epochs, and the validation loss will be displayed every 10 epochs.

It is also possible to load a pre-trained model and continue to train it by passing the address of the epoch folder containing the model checkpoint as the save_addr argument. For example, if the directory for the model at epoch 50 (epoch_50 folder) was passed to save_addr in the example above, then the model would initialise with the epoch 50 parameters and be trained for an additional 50 epochs to reach 100, at which point the model would be saved in a folder called epoch_100 in the same directory as the epoch_50 folder.

The interface for training autoencoders and classifiers is the same.

Loading and testing a trained network
-------------------------------------

Once you have a trained network, it can be loaded and tested out on some hyperspectral data.

To load a trained model on a new dataset, ensure the data has been pre-processed similarly using:

.. code-block:: python

   import data
   new_hypData = data.HypImg( new_img )
   new_hypData.pre_process( 'minmax' )


Then set up the network. The network architecture must be the same as the one used to train the model being loaded. However, this is easy as the directory where models are saved should contain an automatically generated config.json file, which can be used to set up the network with the same architecture:

.. code-block:: python

   net = autoencoder.mlp_1D_network( configFile='model_directory/config.json' )

Once the architecture has been defined, add a model to the network. For example, adding the model that was saved at epoch 100:

.. code-block:: python

   net.add_model( addr='model_directory/epoch_100'), modelName='csa_100' )

Because multiple models can be added to a single network, the added model must be given a name. The name can be anything - the above model is named 'csa_100' because it was trained for 100 epochs using the cosine spectral angle loss function).

When the network is set up and a model has been added, hyperspectral data can be passed through it. To use a trained autoencoder to extract the latent vectors of some spectra:

.. code-block:: python

   dataZ = net.encoder( modelName='csa_100', dataSamples=new_hypData.spectraPrep )

Make sure to refer to the name of the model the network should use. The encoded hyperspectral data (*dataZ*) can also be decoded to get the reconstruction:

.. code-block:: python

   dataY = net.decoder(modelName='csa_100', dataZ=dataZ)

It is also possible to encode and decode in one step with:

.. code-block:: python

   dataY = net.encoder_decoder(modelName='csa_100', dataZ=new_hypData.spectraPrep)


You can use numpy to reshape the latent vector *dataZ* so that it looks like an image again:

.. code-block:: python

   imgZ = numpy.reshape( dataZ, (new_hypData.numRows, new_hypData.numCols, -1) )

Now you should have a basic idea of how to use the **deephyp** toolbox to train an autoencoder for hyperspectral data!