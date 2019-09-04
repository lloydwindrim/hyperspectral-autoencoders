.. deephyp documentation master file, created by
   sphinx-quickstart on Thu Aug 29 19:50:37 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

train multiple models for an MLP
================================

The code block directly below will train several different models for a given MLP (or dense) autoencoder architecture \
on the Pavia Uni hyperspectral dataset. Each model is trained with a different reconstruction loss function. Make sure \
you have a folder in your directory called 'models'. Once trained, look at the next code block to test out the trained \
autoencoder. If you have already downloaded the Pavia Uni dataset (e.g. from another example) you can comment out that \
step.

The network has four encoder and four decoder layers, with 50 neurons in the first layer, 30 in the second, 10 in \
the third and 3 in the fourth layer (the latent layer). Models are trained with 200,000 spectral samples and 100 \
validation samples with a batch size of 1000 samples. Training lasts for 100 epochs, with a learning rate of 0.001 and \
the Adam optimiser. Three different models are trained, each with a different reconstruction loss function: the \
sum-of-squared errors (SSE), cosine spectral angle (CSA) and spectral angle (SA). The train loss and validation loss \
are displayed every 10 epochs. Models are saved at 50 and 100 epochs. The models are saved in the \
models/test_ae_mlp_adv_csa, models/test_ae_mlp_adv_sa and models/test_ae_mlp_adv_sse folders. Note that all of these \
models use the same network object.

.. code-block:: python

   import deephyp.data
   import deephyp.autoencoder

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
    net = deephyp.autoencoder.mlp_1D_network( inputSize=hypData.numBands, encoderSize=[50,30,10,3], activationFunc='relu',
                                      weightInitOpt='truncated_normal', tiedWeights=None, skipConnect=False )

    # setup multiple training operations for the network (with different loss functions)
    net.add_train_op(name='sse', lossFunc='SSE', learning_rate=1e-3, decay_steps=None, decay_rate=None,
                     method='Adam', wd_lambda=0.0)

    net.add_train_op( name='csa', lossFunc='CSA', learning_rate=1e-3, decay_steps=None, decay_rate=None,
                      method='Adam', wd_lambda=0.0 )

    net.add_train_op(name='sa', lossFunc='SA', learning_rate=1e-3, decay_steps=None, decay_rate=None,
                     method='Adam', wd_lambda=0.0)


    # create directories to save the learnt models
    for method in ['sse','csa','sa']:
        model_dir = os.path.join('models','test_ae_mlp_adv_%s'%(method))
        if os.path.exists(model_dir):
            # if directory already exists, delete it
            shutil.rmtree(model_dir)
        os.mkdir(model_dir)

        # train a model for each training op
        dataTrain.reset_batch()
        net.train(dataTrain=dataTrain, dataVal=dataVal, train_op_name=method, n_epochs=100, save_addr=model_dir,
                  visualiseRateTrain=10, visualiseRateVal=10, save_epochs=[50, 100])



The code below will test the MLP (or dense) autoencoder models trained in the above code block, on the Pavia Uni \
hyperspectral dataset. Make sure you have a folder in your directory called 'results'. Run the testing code \
block as a separate script to the training code block. The code block below downloads the Pavia Uni ground truth labels.

The network is setup using the config file output during training. Because all three models use the same network, the \
network can be setup from just one of the config files. Each of the three trained models are added to the network. The \
models are each used to encode a latent representation of the Pavia Uni data and a scatter plot figure of the samples in \
two of the three latent dimensions are shown for each model. The two latent features with the greatest standard \
deviation of the data samples are used for the scatter plot.

.. code-block:: python

   import deephyp.data
   import deephyp.autoencoder

   import scipy.io
   import matplotlib.pyplot as plt
   import os
   import numpy as np
   try:
       from urllib import urlretrieve # python2
   except:
       from urllib.request import urlretrieve # python3


    # read data into numpy array
    mat = scipy.io.loadmat( 'PaviaU.mat' )
    img = mat[ 'paviaU' ]

    # create a hyperspectral dataset object from the numpy array
    hypData = deephyp.data.HypImg( img )

    # pre-process data to make the model easier to train
    hypData.pre_process( 'minmax' )

    # setup a network from a config file
    net = deephyp.autoencoder.mlp_1D_network( configFile=os.path.join('models','test_ae_mlp_adv_sse','config.json') )

    # assign previously trained parameters to the network, and name each model
    net.add_model( addr=os.path.join('models','test_ae_mlp_adv_sse','epoch_100'), modelName='sse_100' )
    net.add_model(addr=os.path.join('models', 'test_ae_mlp_adv_csa', 'epoch_100'), modelName='csa_100')
    net.add_model(addr=os.path.join('models', 'test_ae_mlp_adv_sa', 'epoch_100'), modelName='sa_100')

    # feed forward hyperspectral dataset through each encoder model (get latent encoding)
    dataZ_sse = net.encoder( modelName='sse_100', dataSamples=hypData.spectraPrep )
    dataZ_csa = net.encoder(modelName='csa_100', dataSamples=hypData.spectraPrep)
    dataZ_sa = net.encoder(modelName='sa_100', dataSamples=hypData.spectraPrep)

    # feed forward latent encoding through each decoder model (get reconstruction)
    dataY_sse = net.decoder(modelName='sse_100', dataZ=dataZ_sse)
    dataY_csa = net.decoder(modelName='csa_100', dataZ=dataZ_csa)
    dataY_sa = net.decoder(modelName='sa_100', dataZ=dataZ_sa)


    #--------- visualisation ----------------------------------------

    # download dataset ground truth pixel labels (if already downloaded, comment this out).
    urlretrieve( 'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat',
                            os.path.join(os.getcwd(), 'PaviaU_gt.mat') )

    # read labels into numpy array
    mat_gt = scipy.io.loadmat( 'PaviaU_gt.mat' )
    img_gt = mat_gt['paviaU_gt']
    gt = np.reshape( img_gt , ( -1 ) )

    method = ['sse','csa','sa']

    dataZ_collection = [dataZ_sse, dataZ_csa, dataZ_sa]
    for j,dataZ in enumerate(dataZ_collection):

        # save a scatter plot image of 2 of 3 latent dimensions
        idx = np.argsort(-np.std(dataZ, axis=0))
        fig, ax = plt.subplots()
        for i,gt_class in enumerate(['asphalt', 'meadow', 'gravel','tree','painted metal','bare soil','bitumen','brick','shadow']):
            ax.scatter(dataZ[gt == i+1, idx[0]], dataZ[gt == i+1, idx[1]], c='C%i'%i,s=5,label=gt_class)
        ax.legend()
        plt.title('latent representation: %s'%(method[j]))
        plt.xlabel('latent feature %i' % (idx[0]))
        plt.ylabel('latent feature %i' % (idx[1]))
        plt.savefig(os.path.join('results', 'test_mlp_scatter_%s.png'%(method[j])))



