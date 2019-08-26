'''
    File name: classifier_train_CNN_basic.py
    Author: Lloyd Windrim
    Date created: August 2019
    Python package: deephyp

    Description: An example script for training a CNN classifier on the Pavia Uni hyperspectral dataset.

'''

import scipy.io
import os
import shutil
import numpy as np
from utils import reporthook
try:
    from urllib import urlretrieve # python2
except:
    from urllib.request import urlretrieve # python3

# import toolbox libraries
import sys
sys.path.insert(0, '..')
from deephyp import classifier
from deephyp import data

if __name__ == '__main__':

    # download dataset and ground truth (if already downloaded, comment this out)
    # Use urllib.request.urlretrieve for python3
    urlretrieve( 'http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                        os.path.join(os.getcwd(),'PaviaU.mat'), reporthook )
    urlretrieve('http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat',
                       os.path.join(os.getcwd(), 'PaviaU_gt.mat'), reporthook)

    # read data into numpy array
    mat = scipy.io.loadmat('PaviaU.mat')
    img = mat['paviaU']

    # read labels into numpy array
    mat_gt = scipy.io.loadmat('PaviaU_gt.mat')
    img_gt = mat_gt['paviaU_gt']

    # create a hyperspectral dataset object from the numpy array
    hypData = data.HypImg( img, labels=img_gt )

    # pre-process data to make the model easier to train
    hypData.pre_process( 'minmax' )

    # get indices for training and validation data
    trainSamples = 50 # per class
    valSamples = 15 # per class
    train_indices = []
    for i in range(1,10):
        train_indices += np.nonzero(hypData.labels == i)[0][:trainSamples].tolist()
    val_indices = []
    for i in range(1,10):
        val_indices += np.nonzero(hypData.labels == i)[0][trainSamples:trainSamples+valSamples].tolist()

    # create data iterator objects for training and validation using the pre-processed data
    dataTrain = data.Iterator( dataSamples=hypData.spectraPrep[train_indices, :],
                              targets=hypData.labelsOnehot[train_indices,:], batchSize=50 )
    dataVal = data.Iterator( dataSamples=hypData.spectraPrep[val_indices, :],
                            targets=hypData.labelsOnehot[val_indices,:] )

    # shuffle training data
    dataTrain.shuffle()

    # setup a cnn classifier with 3 convolutional layers and 2 fully-connected layers
    net = classifier.cnn_1D_network( inputSize=hypData.numBands, numClasses=9, convFilterSize=[20,10,10],
                  convNumFilters=[10,10,10], convStride = [1,1,1], fcSize=[20,20], activationFunc='relu',
                  weightInitOpt='truncated_normal', weightStd=0.1, padding='VALID' )

    # setup a training operation
    net.add_train_op('basic50',balance_classes=True)

    # create a directory to save the learnt model
    model_dir = os.path.join('models', 'test_clf')
    if os.path.exists(model_dir):
        # if directory already exists, delete it
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    # train the network for 1000 epochs, saving the model at epoch 100 and 1000
    net.train(dataTrain=dataTrain, dataVal=dataVal, train_op_name='basic50', n_epochs=1000, save_addr=model_dir,
              visualiseRateTrain=10, visualiseRateVal=10, save_epochs=[100,1000])
