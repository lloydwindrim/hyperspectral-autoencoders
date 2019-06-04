import scipy.io
import tensorflow as tf

import data

mat = scipy.io.loadmat('/Users/lloydwindrim/Documents/projects/datasets/hyperspectral/PaviaU.mat')
img = mat['paviaU']

hypData = data.Data(img)



print 'hello'