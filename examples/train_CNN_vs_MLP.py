'''
    File name: train_CNN_vs_MLP.py
    Author: Lloyd Windrim
    Date created: August 2019
    Python package: deephyp

    Description: An example script for training an MLP (or dense) autoencoder and a convolutional autoencoder on the
    Pavia Uni hyperspectral dataset.

'''


import scipy.io
import urllib
import os
import shutil
from utils import reporthook


# import toolbox libraries
import sys
sys.path.insert(0, '..')
from deephyp import autoencoder
from deephyp import data


if __name__ == '__main__':

    # download dataset (if already downloaded, comment this out)
    #urllib.urlretrieve( 'http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat', os.path.join(os.getcwd(),'PaviaU.mat'), reporthook )

    # read data into numpy array
    mat = scipy.io.loadmat( 'PaviaU.mat' )
    img = mat[ 'paviaU' ]

    # create a hyperspectral dataset object from the numpy array
    hypData = data.HypImg( img )

    # pre-process data to make the model easier to train
    hypData.pre_process( 'minmax' )

    # create data iterator objects for training and validation using the pre-processed data
    trainSamples = 200000
    valSamples = 100
    dataTrain = data.Iterator( dataSamples=hypData.spectraPrep[:trainSamples, :],
                              targets=hypData.spectraPrep[:trainSamples, :], batchSize=1000 )
    dataVal = data.Iterator( dataSamples=hypData.spectraPrep[trainSamples:trainSamples+valSamples, :],
                            targets=hypData.spectraPrep[trainSamples:trainSamples+valSamples, :] )

    # shuffle training data
    dataTrain.shuffle()

    # setup a fully-connected autoencoder neural network with 3 encoder layers
    net_mlp = autoencoder.mlp_1D_network( inputSize=hypData.numBands, encoderSize=[50,30,10,3], activationFunc='relu',
                                      weightInitOpt='truncated_normal', tiedWeights=None, skipConnect=False )

    # setup a convolutional autoencoder neural network with 3 conv encoder layers
    net_cnn = autoencoder.cnn_1D_network( inputSize=hypData.numBands, zDim=3, encoderNumFilters=[10,10,10] ,
                                     encoderFilterSize=[20,10,10], activationFunc='relu', weightInitOpt='truncated_normal',
                                     encoderStride=[1, 1, 1], tiedWeights=None, skipConnect=False )

    # setup a training operation for each network (using the same loss function)
    net_mlp.add_train_op(name='csa', lossFunc='CSA', learning_rate=1e-3, decay_steps=None, decay_rate=None,
                     method='Adam', wd_lambda=0.0)

    net_cnn.add_train_op( name='csa', lossFunc='CSA', learning_rate=1e-3, decay_steps=None, decay_rate=None,
                      method='Adam', wd_lambda=0.0 )



    # create directories to save the learnt models
    model_dirs = []
    for method in ['mlp','cnn']:
        model_dir = os.path.join('models','test_comparison_%s'%(method))
        if os.path.exists(model_dir):
            # if directory already exists, delete it
            shutil.rmtree(model_dir)
        os.mkdir(model_dir)
        model_dirs.append( model_dir )

    # train the mlp model (100 epochs)
    dataTrain.reset_batch()
    net_mlp.train(dataTrain=dataTrain, dataVal=dataVal, train_op_name='csa', n_epochs=100, save_addr=model_dirs[0],
              visualiseRateTrain=10, visualiseRateVal=10, save_epochs=[100])

    # train the cnn model (takes longer, so only 10 epochs)
    dataTrain.reset_batch()
    net_cnn.train(dataTrain=dataTrain, dataVal=dataVal, train_op_name='csa', n_epochs=10, save_addr=model_dirs[1],
              visualiseRateTrain=1, visualiseRateVal=10, save_epochs=[10])



