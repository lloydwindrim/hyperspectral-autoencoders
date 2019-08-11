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

    # setup a network from a config file
    net = autoencoder.mlp_1D_network( configFile=os.path.join('models','test_mlp_sid','config.json') )

    # assign previously trained parameters to the network, and name model
    net.add_model( addr=os.path.join('models','test_mlp_sid','epoch_100'), modelName='sid_100' )

    # feed forward hyperspectral dataset through encoder (get latent encoding)
    dataZ = net.encoder( modelName='sid_100', dataSamples=hypData.spectraPrep )

    # feed forward latent encoding through decoder (get reconstruction)
    dataY = net.decoder(modelName='sid_100', dataZ=dataZ)


    #--------- visualisation ----------------------------------------

    # download dataset ground truth pixel labels (if already downloaded, comment this out)
    urllib.urlretrieve( 'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat',
                       os.path.join(os.getcwd(), 'PaviaU_gt.mat'), reporthook )

    # read labels into numpy array
    mat_gt = scipy.io.loadmat( 'PaviaU_gt.mat' )
    img_gt = mat_gt['paviaU_gt']
    gt = np.reshape( img_gt , ( -1 ) )


    # save a scatter plot image of 2 of 3 latent dimensions
    idx = np.argsort(-np.std(dataZ, axis=0))
    fig, ax = plt.subplots()
    for i,gt_class in enumerate(['asphault', 'meadow', 'gravel','tree','painted metal','bare soil','bitumen','brick','shadow']):
        ax.scatter(dataZ[gt == i+1, idx[0]], dataZ[gt == i+1, idx[1]], c='C%i'%i,s=5,label=gt_class)
    ax.legend()
    plt.title('latent representation: sid')
    plt.savefig(os.path.join('results', 'test_mlp_scatter_sid.png'))

