import scipy.io
import tensorflow as tf

import data
import autoencoder

# read data
mat = scipy.io.loadmat('/Users/lloydwindrim/Documents/projects/datasets/hyperspectral/PaviaU.mat')
img = mat['paviaU']

# create data

hypData = data.Img(img)

# pre-process
hypData.pre_process('minmax')

# setup network
net = autoencoder.mlp_1D_network( hypData.numBands, activationFunc='sigmoid',tiedWeights=None, skipConnect=False )

# create training and validation data objects
dataTrain = data.Iterator(hypData.spectraPrep[:200000,:],targets=hypData.spectraPrep[:200000,:],batchSize=1000)
dataTrain.shuffle()
dataVal = data.Iterator(hypData.spectraPrep[200000:200100,:],targets=hypData.spectraPrep[200000:200100,:])

# train network
save_addr = 'models/test_model'
net.add_train_op('sse',lossFunc='CSA',learning_rate=1e-3,wd_lambda=0.0)
net.train( dataTrain, dataVal, 'sse', 100, save_addr, visualiseRateTrain=10, visualiseRateVal=10, save_epochs=[100] )


# encode entire image



print 'hello'