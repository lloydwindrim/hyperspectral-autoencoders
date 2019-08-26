'''
    File name: classifier_test_CNN_basic.py
    Author: Lloyd Windrim
    Date created: August 2019
    Python package: deephyp

    Description: An example script for testing a trained CNN classifier on the Pavia Uni hyperspectral dataset. Saves
    a figure showing the ground truth class labels and separate figures showing the predicted class labels with and
    without the background masked out.

'''

import scipy.io
import urllib
import os
import shutil
import numpy as np
import pylab as pl

# import toolbox libraries
import sys
sys.path.insert(0, '..')
from deephyp import classifier
from deephyp import data

if __name__ == '__main__':

    # read data into numpy array
    mat = scipy.io.loadmat('PaviaU.mat')
    img = mat['paviaU']

    # create a hyperspectral dataset object from the numpy array
    hypData = data.HypImg( img )

    # pre-process data to make the model easier to train
    hypData.pre_process('minmax')


    # setup a fully-connected autoencoder neural network with 3 encoder layers
    net = classifier.cnn_1D_network(configFile=os.path.join('models','test_clf_cnn','config.json'))

    # assign previously trained parameters to the network, and name model
    net.add_model( addr=os.path.join('models','test_clf_cnn','epoch_1000'), modelName='basic_model' )

    # feed forward hyperspectral dataset through the model to predict class labels and scores for each sample
    data_pred = net.predict_labels( modelName='basic_model', dataSamples=hypData.spectraPrep  )
    data_scores = net.predict_scores( modelName='basic_model', dataSamples=hypData.spectraPrep  )

    # extract features at second last layer
    data_features = net.predict_features(modelName='basic_model', dataSamples=hypData.spectraPrep, layer=net.numLayers-1)

    #--------- visualisation ----------------------------------------

    # reshape predicted labels to an image
    img_pred = np.reshape(data_pred, (hypData.numRows, hypData.numCols))

    # read labels into numpy array
    mat_gt = scipy.io.loadmat('PaviaU_gt.mat')
    img_gt = mat_gt['paviaU_gt']


    class_names = ['asphault', 'meadow', 'gravel','tree','painted metal','bare soil','bitumen','brick','shadow']
    cmap = pl.cm.jet

    # save ground truth figure
    pl.figure()
    for entry in pl.unique(img_gt):
        colour = cmap(entry*255/(np.max(img_gt) - 0))
        pl.plot(0, 0, "-", c=colour, label=(['background']+class_names)[entry])
    pl.imshow(img_gt,cmap=cmap)
    pl.legend(bbox_to_anchor=(2, 1))
    pl.title('ground truth labels')
    pl.savefig(os.path.join('results', 'test_classification_gt.png'))

    # save predicted classes figure
    pl.figure()
    for entry in pl.unique(img_pred):
        colour = cmap(entry*255/(np.max(img_pred) - 0))
        pl.plot(0, 0, "-", c=colour, label=class_names[entry-1])
    pl.imshow(img_pred,cmap=cmap)
    pl.legend(bbox_to_anchor=(2, 1))
    pl.title('classification prediction')
    pl.savefig(os.path.join('results', 'test_classification_pred.png'))

    # save predicted classes figure with background masked out
    img_pred[img_gt==0] = 0
    pl.figure()
    for entry in pl.unique(img_pred):
        colour = cmap(entry*255/(np.max(img_pred) - 0))
        pl.plot(0, 0, "-", c=colour, label=(['background']+class_names)[entry])
    pl.imshow(img_pred,cmap=cmap)
    pl.legend(bbox_to_anchor=(2, 1))
    pl.title('classification prediction with background masked')
    pl.savefig(os.path.join('results', 'test_classification_pred_bkgrd.png'))



