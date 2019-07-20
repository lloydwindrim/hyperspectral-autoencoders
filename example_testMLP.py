import scipy.io
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

import data
import autoencoder

# read data
mat = scipy.io.loadmat('/Users/lloydwindrim/Documents/projects/datasets/hyperspectral/PaviaU.mat')
img = mat['paviaU']

# create data object
hypData = data.Img(img)

# pre-process
hypData.pre_process('minmax')

# setup network
net = autoencoder.mlp_1D_network( hypData.numBands, activationFunc='sigmoid',tiedWeights=None, skipConnect=False )
net.add_model('models/test_model/epoch_100','sse_500')


# import network_ops
# x1 = tf.placeholder("float", [None,103])
# x2 = tf.placeholder("float", [None,103])
#
# a = hypData.spectraPrep[[0,1],:]
# a = hypData.spectraPrep[:200000,:]
#
# sess = tf.Session()
#
# loss = network_ops.loss_function_reconstruction_1D(x1,x2,'SID')
# print sess.run(loss,feed_dict={x1:a,x2:a})



# encode entire image
dataY = net.encoder_decoder( 'sse_500', hypData.spectraPrep  )
img = np.reshape(dataY,(hypData.numRows,hypData.numCols,hypData.numBands))

print 'hello'