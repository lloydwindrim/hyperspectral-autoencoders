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

# An example script for testing several different trained models for a given MLP (or dense) autoencoder architecture
# using the Pavia Uni hyperspectral dataset.

if __name__ == '__main__':

    # read data into numpy array
    mat = scipy.io.loadmat( 'PaviaU.mat' )
    img = mat[ 'paviaU' ]

    # create a hyperspectral dataset object from the numpy array
    hypData = data.HypImg( img )

    # pre-process data to make the model easier to train
    hypData.pre_process( 'minmax' )

    # setup a network from a config file
    net = autoencoder.mlp_1D_network( configFile=os.path.join('models','test_mlp_adv_sse','config.json') )

    # assign previously trained parameters to the network, and name each model
    net.add_model( addr=os.path.join('models','test_mlp_adv_sse','epoch_100'), modelName='sse_100' )
    net.add_model(addr=os.path.join('models', 'test_mlp_adv_csa', 'epoch_100'), modelName='csa_100')
    net.add_model(addr=os.path.join('models', 'test_mlp_adv_sa', 'epoch_100'), modelName='sa_100')

    # feed forward hyperspectral dataset through each encoder model (get latent encoding)
    dataZ_sse = net.encoder( modelName='sse_100', dataSamples=hypData.spectraPrep )
    dataZ_csa = net.encoder(modelName='csa_100', dataSamples=hypData.spectraPrep)
    dataZ_sa = net.encoder(modelName='sa_100', dataSamples=hypData.spectraPrep)

    # feed forward latent encoding through each decoder model (get reconstruction)
    dataY_sse = net.decoder(modelName='sse_100', dataZ=dataZ_sse)
    dataY_csa = net.decoder(modelName='csa_100', dataZ=dataZ_csa)
    dataY_sa = net.decoder(modelName='sa_100', dataZ=dataZ_sa)


    #--------- visualisation ----------------------------------------

    # download dataset ground truth pixel labels (if already downloaded, comment this out)
    urllib.urlretrieve( 'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat',
                       os.path.join(os.getcwd(), 'PaviaU_gt.mat'), reporthook )

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
        for i,gt_class in enumerate(['asphault', 'meadow', 'gravel','tree','painted metal','bare soil','bitumen','brick','shadow']):
            ax.scatter(dataZ[gt == i+1, idx[0]], dataZ[gt == i+1, idx[1]], c='C%i'%i,s=5,label=gt_class)
        ax.legend()
        plt.title('latent representation: %s'%(method[j]))
        plt.savefig(os.path.join('results', 'test_mlp_scatter_%s.png'%(method[j])))

