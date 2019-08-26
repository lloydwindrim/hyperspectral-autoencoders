'''
    File name: autoencoder_test_MLP_basic.py
    Author: Lloyd Windrim
    Date created: August 2019
    Python package: deephyp

    Description: An example script for testing a trained MLP (or dense) autoencoder on the Pavia Uni hyperspectral
    dataset. Saves a figure of the latent vector for a 'meadow' spectral sample and the reconstruction.

'''

import scipy.io
import matplotlib.pyplot as plt
import os
import numpy as np


# import toolbox libraries
import sys
sys.path.insert(0, '..')
from deephyp import autoencoder
from deephyp import data



if __name__ == '__main__':

    # read data into numpy array
    mat = scipy.io.loadmat( 'PaviaU.mat' )
    img = mat[ 'paviaU' ]

    # create a hyperspectral dataset object from the numpy array
    hypData = data.HypImg( img )

    # pre-process data to make the model easier to train
    hypData.pre_process( 'minmax' )

    # setup a network from a config file
    net = autoencoder.mlp_1D_network( configFile=os.path.join('models','test_ae_mlp','config.json') )

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

