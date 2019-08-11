import scipy.io
import matplotlib.pyplot as plt
import os
import numpy as np
import urllib
from utils import reporthook


# import toolbox libraries
import sys
sys.path.insert(0, os.path.join('..','toolbox'))
import autoencoder
import data

if __name__ == '__main__':

    # read data into numpy array
    mat = scipy.io.loadmat( 'PaviaU.mat' )
    img = mat[ 'paviaU' ]

    # create a hyperspectral dataset object from the numpy array
    hypData = data.HypImg( img )

    # pre-process data to make the model easier to train
    hypData.pre_process( 'minmax' )

    # setup each network from the config files
    net_mlp = autoencoder.mlp_1D_network( configFile=os.path.join('models','test_comparison_mlp','config.json') )
    net_cnn = autoencoder.cnn_1D_network(configFile=os.path.join('models', 'test_comparison_cnn', 'config.json'))

    # assign previously trained parameters to the network, and name each model
    net_mlp.add_model( addr=os.path.join('models','test_comparison_mlp','epoch_100'), modelName='mlp_100' )
    net_cnn.add_model(addr=os.path.join('models', 'test_comparison_cnn', 'epoch_10'), modelName='cnn_10')


    # feed forward hyperspectral dataset through each encoder model (get latent encoding)
    dataZ_mlp = net_mlp.encoder( modelName='mlp_100', dataSamples=hypData.spectraPrep )
    dataZ_cnn = net_cnn.encoder(modelName='cnn_10', dataSamples=hypData.spectraPrep)


    # feed forward latent encoding through each decoder model (get reconstruction)
    dataY_mlp = net_mlp.decoder(modelName='mlp_100', dataZ=dataZ_mlp)
    dataY_cnn = net_cnn.decoder(modelName='cnn_10', dataZ=dataZ_cnn)



    #--------- visualisation ----------------------------------------

    # download dataset ground truth pixel labels (if already downloaded, comment this out)
    urllib.urlretrieve( 'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat',
                       os.path.join(os.getcwd(), 'PaviaU_gt.mat'), reporthook )

    # read labels into numpy array
    mat_gt = scipy.io.loadmat( 'PaviaU_gt.mat' )
    img_gt = mat_gt['paviaU_gt']
    gt = np.reshape( img_gt , ( -1 ) )

    method = ['mlp','cnn']

    dataZ_collection = [dataZ_mlp, dataZ_cnn]
    for j,dataZ in enumerate(dataZ_collection):

        # save a scatter plot image of 2 of 3 latent dimensions
        idx = np.argsort(-np.std(dataZ, axis=0))
        fig, ax = plt.subplots()
        for i,gt_class in enumerate(['asphault', 'meadow', 'gravel','tree','painted metal','bare soil','bitumen','brick','shadow']):
            ax.scatter(dataZ[gt == i+1, idx[0]], dataZ[gt == i+1, idx[1]], c='C%i'%i,s=5,label=gt_class)
        ax.legend()
        plt.title('latent representation: %s'%(method[j]))
        plt.savefig(os.path.join('results', 'test_comparison_%s.png'%(method[j])))


    # reshape reconstruction to original image dimensions
    imgY_mlp = np.reshape(dataY_mlp, (hypData.numRows, hypData.numCols, -1))
    imgY_cnn = np.reshape(dataY_cnn, (hypData.numRows, hypData.numCols, -1))
    imgX = np.reshape(hypData.spectraPrep, (hypData.numRows, hypData.numCols, -1))

    # save plot comparing pre-processed 'meadow' spectra input with decoder reconstruction
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(range(hypData.numBands),imgX[576, 210, :],label='pre-processed input')
    ax.plot(range(hypData.numBands),imgY_mlp[576, 210, :],label='mlp reconstruction')
    ax.plot(range(hypData.numBands), imgY_cnn[576, 210, :], label='cnn reconstruction')
    plt.xlabel('band')
    plt.ylabel('value')
    plt.title('meadow spectra')
    ax.legend()
    plt.savefig(os.path.join('results', 'test_reconstruct_comparison.png'))