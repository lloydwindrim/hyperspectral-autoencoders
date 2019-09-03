.. deephyp documentation master file, created by
   sphinx-quickstart on Thu Aug 29 19:50:37 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

train an MLP with the SID loss function
=======================================

The code block directly below will train an MLP (or dense) autoencoder on the Pavia Uni hyperspectral dataset. Make sure \
you have a folder in your directory called 'models'. Once trained, look at the next code block to test out the trained \
autoencoder. If you have already downloaded the Pavia Uni dataset (e.g. from another example) you can comment out that \
step.

The network has three encoder and three decoder layers, with 50 neurons in the first layer, 30 in the second and 10 in \
the third (the latent layer). A model is trained with 200,000 spectral samples and 100 validation samples with a batch \
size of 1000 samples. Training lasts for 100 epochs, with a learning rate of 0.001, the Adam optimiser and spectral \
information divergence (SID) reconstruction loss function. The train loss and validation loss are displayed every 10 \
epochs. Models are saved at 50 and 100 epochs. The models are saved in the models/test_ae_mlp_sid folder.

Since the SID loss contains log in its expression which is undefined for values <= 0, it is best to use sigmoid as the \
activation function (including the final activation function) for networks trained with the SID loss.

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
    net = deephyp.autoencoder.mlp_1D_network( inputSize=hypData.numBands, encoderSize=[50,30,10], activationFunc='sigmoid',
                                      weightInitOpt='truncated_normal', tiedWeights=None, skipConnect=False,
                                      activationFuncFinal='sigmoid' )

    # setup a training operation for the network
    net.add_train_op( name='sid', lossFunc='SID', learning_rate=1e-3, decay_steps=None, decay_rate=None,
                      method='Adam', wd_lambda=0.0 )

    # create a directory to save the learnt model
    model_dir = os.path.join('models','test_ae_mlp_sid')
    if os.path.exists(model_dir):
        # if directory already exists, delete it
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    # train the network for 100 epochs, saving the model at epoch 50 and 100
    net.train(dataTrain=dataTrain, dataVal=dataVal, train_op_name='sid', n_epochs=100, save_addr=model_dir,
              visualiseRateTrain=10, visualiseRateVal=10, save_epochs=[100])




The code below will test a trained MLP (or dense) autoencoder on the Pavia Uni hyperspectral dataset. Make sure you have \
a folder in your directory called 'results'. The network can be trained using the above code block. Run the testing code \
block as a separate script to the training code block.

The network is setup using the config file output during training. Then the 100 epoch model is added (named 'sid_100'). \
The model is used to encode a latent representation of the Pavia Uni data and a scatter plot figure of the samples in \
two of the ten latent dimensions are shown for each model. The two latent features with the greatest standard deviation \
of the data samples are used for the scatter plot.

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
    net = deephyp.autoencoder.mlp_1D_network( configFile=os.path.join('models','test_ae_mlp_sid','config.json') )

    # assign previously trained parameters to the network, and name model
    net.add_model( addr=os.path.join('models','test_ae_mlp_sid','epoch_100'), modelName='sid_100' )

    # feed forward hyperspectral dataset through encoder (get latent encoding)
    dataZ = net.encoder( modelName='sid_100', dataSamples=hypData.spectraPrep )

    # feed forward latent encoding through decoder (get reconstruction)
    dataY = net.decoder(modelName='sid_100', dataZ=dataZ)


    #--------- visualisation ----------------------------------------

    # download dataset ground truth pixel labels (if already downloaded, comment this out)
    urlretrieve( 'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat',
                       os.path.join(os.getcwd(), 'PaviaU_gt.mat') )

    # read labels into numpy array
    mat_gt = scipy.io.loadmat( 'PaviaU_gt.mat' )
    img_gt = mat_gt['paviaU_gt']
    gt = np.reshape( img_gt , ( -1 ) )


    # save a scatter plot image of 2 of 3 latent dimensions
    idx = np.argsort(-np.std(dataZ, axis=0))
    fig, ax = plt.subplots()
    for i,gt_class in enumerate(['asphalt', 'meadow', 'gravel','tree','painted metal','bare soil','bitumen','brick','shadow']):
        ax.scatter(dataZ[gt == i+1, idx[0]], dataZ[gt == i+1, idx[1]], c='C%i'%i,s=5,label=gt_class)
    ax.legend()
    plt.title('latent representation: sid')
    plt.xlabel('latent feature %i' % (idx[0]))
    plt.ylabel('latent feature %i' % (idx[1]))
    plt.savefig(os.path.join('results', 'test_mlp_scatter_sid.png'))


