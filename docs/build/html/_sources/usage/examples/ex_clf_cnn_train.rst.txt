.. deephyp documentation master file, created by
   sphinx-quickstart on Thu Aug 29 19:50:37 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

train and test a CNN classifier
===============================

The code block directly below will train CNN classifier on the Pavia Uni hyperspectral dataset. Make sure you have a \
folder in your directory called 'models'. Once trained, look at the next code block to test out the trained classifier. \
If you have already downloaded the Pavia Uni dataset and ground truth dataset (e.g. from another example) you can \
comment out that step.

The CNN classification network has three convolutional layers and three fully-connected layers (including the output \
layer). The first convolutional layer has 10 filters of size 20, with the second and third both having \
10 filters of size 10. All convolutional layers have a stride of 1. The first two fully-connected layers both have 20 \
neurons and the final fully-connected layer has 9 neurons (because there are 9 classes). A ReLU activation function is \
used.

The CNN model is trained on 50 samples per each of the 9 classes (not including the background class, which has a label \
of zero). 15 samples per class are used for validation, with a batch size of 50. The network is trained for 1000 epochs \
using the cross-entropy loss function with class balancing (even though the number of samples per class is already \
balanced). Both the train and validation loss are visualised every 10 epochs and models are saved at epochs 100 and 1000. \
The models are saved in the models/test_clf_cnn folders.


.. code-block:: python

   import deephyp

   import scipy.io
   import os
   import shutil
   try:
       from urllib import urlretrieve # python2
   except:
       from urllib.request import urlretrieve # python3


    # download dataset and ground truth (if already downloaded, comment this out)
    urlretrieve( 'http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat', os.path.join(os.getcwd(),'PaviaU.mat') )
    urlretrieve('http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat', os.path.join(os.getcwd(), 'PaviaU_gt.mat') )

    # read data into numpy array
    mat = scipy.io.loadmat('PaviaU.mat')
    img = mat['paviaU']

    # read labels into numpy array
    mat_gt = scipy.io.loadmat('PaviaU_gt.mat')
    img_gt = mat_gt['paviaU_gt']

    # create a hyperspectral dataset object from the numpy array
    hypData = deephyp.data.HypImg( img, labels=img_gt )

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
    dataTrain = deephyp.data.Iterator( dataSamples=hypData.spectraPrep[train_indices, :],
                              targets=hypData.labelsOnehot[train_indices,:], batchSize=50 )
    dataVal = deephyp.data.Iterator( dataSamples=hypData.spectraPrep[val_indices, :],
                            targets=hypData.labelsOnehot[val_indices,:] )

    # shuffle training data
    dataTrain.shuffle()

    # setup a cnn classifier with 3 convolutional layers and 2 fully-connected layers
    net = deephyp.classifier.cnn_1D_network( inputSize=hypData.numBands, numClasses=9, convFilterSize=[20,10,10],
                  convNumFilters=[10,10,10], convStride = [1,1,1], fcSize=[20,20], activationFunc='relu',
                  weightInitOpt='truncated_normal', weightStd=0.1, padding='VALID' )

    # setup a training operation
    net.add_train_op('basic50',balance_classes=True)

    # create a directory to save the learnt model
    model_dir = os.path.join('models', 'test_clf_cnn')
    if os.path.exists(model_dir):
        # if directory already exists, delete it
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    # train the network for 1000 epochs, saving the model at epoch 100 and 1000
    net.train(dataTrain=dataTrain, dataVal=dataVal, train_op_name='basic50', n_epochs=1000, save_addr=model_dir,
              visualiseRateTrain=10, visualiseRateVal=10, save_epochs=[100,1000])


The code below will test the CNN classifier model trained in the above code block, on the Pavia Uni \
hyperspectral dataset. Make sure you have a folder in your directory called 'results'. Run the testing code \
block as a separate script to the training code block.

The network is setup using the config file output during training. The model is added to the network (with the name \
'basic_model'). Pavia Uni data samples from the entire image are passed through the network, which predicts labels and \
class labels and scores for each sample. Figures are saved showing the predicted class labels for the image with and \
without the background class masked out, as well as showing the ground truth labels.

.. code-block:: python

   import deephyp

   import scipy.io
   import pylab as pl
   import os
   import numpy as np


    # read data into numpy array
    mat = scipy.io.loadmat('PaviaU.mat')
    img = mat['paviaU']

    # create a hyperspectral dataset object from the numpy array
    hypData = deephyp.data.HypImg( img )

    # pre-process data to make the model easier to train
    hypData.pre_process('minmax')


    # setup a fully-connected autoencoder neural network with 3 encoder layers
    net = deephyp.classifier.cnn_1D_network(configFile=os.path.join('models','test_clf_cnn','config.json'))

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


    class_names = ['asphalt', 'meadow', 'gravel','tree','painted metal','bare soil','bitumen','brick','shadow']
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


