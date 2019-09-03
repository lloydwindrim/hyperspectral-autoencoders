.. deephyp documentation master file, created by
   sphinx-quickstart on Thu Aug 29 19:50:37 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

train and test a basic MLP
==========================

The code block directly below will train an MLP (or dense) autoencoder on the Pavia Uni hyperspectral dataset. Make sure \
you have a folder in your directory called 'models'. Once trained, look at the next code block to test out the trained \
autoencoder. If you have already downloaded the Pavia Uni dataset (e.g. from another example) you can comment out that \
step.

The network has three encoder and three decoder layers, with 50 neurons in the first layer, 30 in the second and 10 in \
the third (the latent layer). A model is trained with 200,000 spectral samples and 100 validation samples with a batch \
size of 1000 samples. Training lasts for 100 epochs, with a learning rate of 0.001, the Adam optimiser and cosine \
spectral angle (CSA) reconstruction loss function. The train loss and validation loss are displayed every 10 epochs. \
Models are saved at 50 and 100 epochs. The models are saved in the models/test_ae_mlp folder.

.. code-block:: python

   import deephyp

   import scipy.io
   import os
   import shutil
   try:
       from urllib import urlretrieve # python2
   except:
       from urllib.request import urlretrieve # python3


    # download dataset (if already downloaded, comment this out)
    urlretrieve( 'http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat', os.path.join(os.getcwd(),'PaviaU.mat') )

    # read data into numpy array
    mat = scipy.io.loadmat( 'PaviaU.mat' )
    img = mat[ 'paviaU' ]

    # create a hyperspectral dataset object from the numpy array
    hypData = deephyp.data.HypImg( img )

    # pre-process data to make the model easier to train
    hypData.pre_process( 'minmax' )

    # create data iterator objects for training and validation using the pre-processed data
    trainSamples = 200000
    valSamples = 100
    dataTrain = deephyp.data.Iterator( dataSamples=hypData.spectraPrep[:trainSamples, :],
                              targets=hypData.spectraPrep[:trainSamples, :], batchSize=1000 )
    dataVal = deephyp.data.Iterator( dataSamples=hypData.spectraPrep[trainSamples:trainSamples+valSamples, :],
                            targets=hypData.spectraPrep[trainSamples:trainSamples+valSamples, :] )

    # shuffle training data
    dataTrain.shuffle()

    # setup a fully-connected autoencoder neural network with 3 encoder layers
    net = deephyp.autoencoder.mlp_1D_network( inputSize=hypData.numBands, encoderSize=[50,30,10], activationFunc='relu',
                                      weightInitOpt='truncated_normal', tiedWeights=None, skipConnect=False )

    # setup a training operation for the network
    net.add_train_op( name='csa', lossFunc='CSA', learning_rate=1e-3, decay_steps=None, decay_rate=None,
                      method='Adam', wd_lambda=0.0 )

    # create a directory to save the learnt model
    model_dir = os.path.join('models','test_ae_mlp')
    if os.path.exists(model_dir):
        # if directory already exists, delete it
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    # train the network for 100 epochs, saving the model at epoch 50 and 100
    net.train(dataTrain=dataTrain, dataVal=dataVal, train_op_name='csa', n_epochs=100, save_addr=model_dir,
              visualiseRateTrain=10, visualiseRateVal=10, save_epochs=[50,100])




The code below will test a trained MLP (or dense) autoencoder on the Pavia Uni hyperspectral dataset. Make sure you have \
a folder in your directory called 'results'. The network can be trained using the above code block. Run the testing code \
block as a separate script to the training code block.

The network is setup using the config file output during training. Then the 100 epoch model is added (named 'csa_100'). \
The model is used to encode a latent representation of the Pavia Uni data, and reconstruct it from the latent \
representation. A figure of the latent vector for a 'meadow' spectral sample and the reconstruction is saved in the \
results folder.

.. code-block:: python

   import deephyp

   import scipy.io
   import matplotlib.pyplot as plt
   import os
   import numpy as np


    # read data into numpy array
    mat = scipy.io.loadmat( 'PaviaU.mat' )
    img = mat[ 'paviaU' ]

    # create a hyperspectral dataset object from the numpy array
    hypData = deephyp.data.HypImg( img )

    # pre-process data to make the model easier to train
    hypData.pre_process( 'minmax' )

    # setup a network from a config file
    net = deephyp.autoencoder.mlp_1D_network( configFile=os.path.join('models','test_ae_mlp','config.json') )

    # assign previously trained parameters to the network, and name model
    net.add_model( addr=os.path.join('models','test_ae_mlp','epoch_100'), modelName='csa_100' )

    # feed forward hyperspectral dataset through encoder (get latent encoding)
    dataZ = net.encoder( modelName='csa_100', dataSamples=hypData.spectraPrep )

    # feed forward latent encoding through decoder (get reconstruction)
    dataY = net.decoder(modelName='csa_100', dataZ=dataZ)


    #--------- visualisation ----------------------------------------

    # reshape latent encoding to original image dimensions
    imgZ = np.reshape(dataZ, (hypData.numRows, hypData.numCols, -1))

    # reshape reconstructed output of decoder
    imgY = np.reshape(dataY, (hypData.numRows, hypData.numCols, -1))

    # reshape pre-processed input
    imgX = np.reshape(hypData.spectraPrep, (hypData.numRows, hypData.numCols, -1))

    # visualise latent image using 3 out of the 10 dimensions
    colourImg = imgZ.copy()
    colourImg = colourImg[ :,:,np.argsort(-np.std(np.std(colourImg, axis=0), axis=0))[:3] ]
    colourImg /= np.max(np.max(colourImg, axis=0), axis=0)

    # save a latent image (using 3 out of the 10 dimensions)
    plt.imsave(os.path.join('results', 'test_mlp_latentImg.png'), colourImg)

    # save plot of latent vector of 'meadow' spectra
    fig = plt.figure()
    plt.plot(imgZ[576, 210, :])
    plt.xlabel('latent dimension')
    plt.ylabel('latent value')
    plt.title('meadow spectra')
    plt.savefig(os.path.join('results', 'test_mlp_latentVector.png'))

    # save plot comparing pre-processed 'meadow' spectra input with decoder reconstruction
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(range(hypData.numBands),imgX[576, 210, :],label='pre-processed input')
    ax.plot(range(hypData.numBands),imgY[576, 210, :],label='reconstruction')
    plt.xlabel('band')
    plt.ylabel('value')
    plt.title('meadow spectra')
    ax.legend()
    plt.savefig(os.path.join('results', 'test_mlp_InputVsReconstruct.png'))



