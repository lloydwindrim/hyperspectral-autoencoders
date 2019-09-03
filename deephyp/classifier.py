'''
    Description: high-level deep learning classes for building, training and using supervised neural network \
    classifiers. Uses functions from the low-level network_ops module.

    - File name: classifier.py
    - Author: Lloyd Windrim
    - Date created: August 2019
    - Python package: deephyp


'''

import tensorflow as tf
from deephyp import network_ops as net_ops


class cnn_1D_network():
    """ Class for setting up a 1-D convolutional neural network (cnn) for classification. Contains several convolutional \
        layers followed by several fully-connected layers. The network outputs scores for each class, for a given set \
        of input data samples.

    Args:
        configFile (str): Optional way of setting up the network. All other inputs can be ignored (will be overwritten). \
                        Pass the address of the .json config file.
        inputSize (int): Number of dimensions of input data (i.e. number of spectral bands). Value must be input if not \
                        using a config file.
        numClasses (int): Number of labelled classes in the dataset (not including the zero class).
        convFilterSize (int list): Size of filter at each convolutional layer. List length is number of \
                        convolutional layers.
        convNumFilters (int list): Number of filters at each convolutional layer of the network. List length is \
                        number of convolutional layers.
        convStride (int list): Stride at each convolutional layer. List length is number of convolutional layers. \
        fcSize (int list): Number of nodes at each fully-connected (i.e. dense) layer of the encoder. List length \
                        is number of fully-connected layers.
        activationFunc (str): Activation function for all layers except the last one. Current options: ['sigmoid', \
                        'relu', 'linear'].
        weightInitOpt (string): Method of weight initialisation. Current options: ['gaussian', 'truncated_normal', \
                    'xavier', 'xavier_improved'].
        weightStd (float): Used by 'gaussian' and 'truncated_normal' weight initialisation methods.
        padding (str): Type of padding used. Current options: ['VALID', 'SAME'].

    Attributes:
        inputSize (int): Number of dimensions of input data (i.e. number of spectral bands).
        activationFunc (str): Activation function for all layers except the last one.
        weightInitOpt (string): Method of weight initialisation.
        weightStd (float): Parameter for 'gaussian' and 'truncated_normal' weight initialisation methods.
        convFilterSize (int list): Size of filter at each convolutional layer. List length is number of \
                        convolutional layers.
        convNumFilters (int list): Number of filters at each convolutional layer of the network. List length is \
                        number of convolutional layers.
        convStride (int list): Stride at each convolutional layer. List length is number of convolutional layers. \
        padding (str): Type of padding used. Current options: ['VALID', 'SAME'].
        fcSize (int list): Number of nodes at each fully-connected (i.e. dense) layer of the encoder. List length \
                        is number of fully-connected layers.
        numLayers (int): Total number of layers (convolutional and fully-connected).
        y_pred (tensor): Output of network - class scores with shape [numSamples x numClasses]. Accessible through the \
            *predict_scores* class functions, requiring a trained model.
        train_ops (dict): Dictionary of names of train and loss ops (suffixed with _train and _loss) added to the \
            network using the *add_train_op* class function. The name (without suffix) is passed to the *train* class \
            function to train the network with the referenced train and loss op.
        modelsAddrs (dict): Dictionary of model names added to the network using the *add_model* class function. The \
            names reference models which can be used by the *predict_scores*, *predict_labels* and *predict_features* \
            class functions.

    """

    def __init__( self , configFile=None, inputSize=None, numClasses=None, convFilterSize=[20,10,10],
                  convNumFilters=[10,10,10], convStride = [1,1,1], fcSize=[20,20], activationFunc='relu',
                  weightInitOpt='truncated_normal', weightStd=0.1, padding='VALID'):

        self.inputSize = inputSize
        self.numClasses = numClasses
        self.activationFunc = activationFunc
        self.weightInitOpt = weightInitOpt
        self.weightStd = weightStd
        self.convFilterSize = convFilterSize
        self.convNumFilters = convNumFilters
        self.convStride = convStride
        self.padding = padding
        self.fcSize = fcSize


        self.net_config = ['inputSize','numClasses','activationFunc','weightInitOpt','weightStd','convFilterSize',
                           'convNumFilters','convStride','padding','fcSize']
        # loading config file overwrites input arguments
        if configFile is not None:
            net_ops.load_config(self,configFile)

        if self.inputSize is None:
            raise Exception('value must be given for inputSize (not None)')

        if self.numClasses is None:
            raise Exception('value must be given for numClasses (not None)')

        if not (len(self.convFilterSize) == len(self.convNumFilters) == len(self.convStride)):
            raise Exception('the length of convNumfilters, convFilterSize and convStride must be equal.')

        self.x = tf.placeholder("float", [None, self.inputSize])
        self.y_target = tf.placeholder("float", [None, self.numClasses])

        self.weights = { }
        self.biases = { }
        self.h = {}
        self.a = {}
        self.train_ops = {}
        self.modelsAddrs = {}

        # pre-compute shape of data after each layer
        self.convDataShape = [self.inputSize]
        for layerNum in range( len( self.convNumFilters )   ):
            self.convDataShape.append( net_ops.conv_output_shape(
                self.convDataShape[layerNum],self.convFilterSize[layerNum],self.padding,self.convStride[layerNum]) )
        self.convDataShape[layerNum + 1] = self.convDataShape[-1] * self.convNumFilters[layerNum]
        self.fcDataShape = []
        for layerNum in range( len( self.fcSize )  ):
            self.fcDataShape.append( self.fcSize[layerNum] )
        self.fcDataShape.append( self.numClasses )


        # conv layer weights
        for layerNum in range( len( self.convNumFilters ) ):
            self.weights['conv_w%i'%(layerNum+1)] = \
                net_ops.create_variable([self.convFilterSize[layerNum], ([1]+self.convNumFilters)[layerNum],
                                         ([1] + self.convNumFilters)[layerNum+1]],weightInitOpt, wd=True)

        # fc layer weights
        self.weights['fc_w1'] = net_ops.create_variable(
            [self.convDataShape[layerNum+1], self.fcSize[0]],self.weightInitOpt, wd=True)
        for layerNum in range( len( self.fcSize ) - 1 ):
            self.weights['fc_w%i'%(layerNum+2)] = \
                net_ops.create_variable([self.fcSize[layerNum],
                                         self.fcSize[layerNum+1]],self.weightInitOpt, wd=True)
        self.weights['fc_w%i'%(layerNum+3)] = net_ops.create_variable(
            [self.fcSize[layerNum+1], self.numClasses], self.weightInitOpt, wd=True)



        # conv layer biases
        for layerNum in range( len( self.convNumFilters )  ):
            self.biases['conv_b%i'%(layerNum+1)] = \
                net_ops.create_variable([self.convNumFilters[layerNum]] , self.weightInitOpt, wd=True)

        # fc layer biases
        for layerNum in range( len( self.fcSize ) ):
            self.biases['fc_b%i'%(layerNum+1)] = \
                net_ops.create_variable([self.fcSize[layerNum]] , self.weightInitOpt, wd=True)
        self.biases['fc_b%i' % (layerNum + 2)] = \
            net_ops.create_variable([self.numClasses], self.weightInitOpt, wd=True)



        # build network using conv layers, fc layers and x placeholder as input
        self.a['a0'] = tf.expand_dims(self.x,axis=2)   # expand to shape None x inputSize x 1

        # conv layers
        for layerNum in range( 1 , len( self.convNumFilters ) + 1 ):
            self.h['h%d' % (layerNum)] = \
                net_ops.layer_conv1d(self.a['a%d'%(layerNum-1)], self.weights['conv_w%d'%(layerNum)],
                                     self.biases['conv_b%d'%(layerNum)],padding=self.padding,stride=self.convStride[layerNum-1])
            self.a['a%d' % (layerNum)] = net_ops.layer_activation(self.h['h%d' % (layerNum)], self.activationFunc)
        self.a['a%d'%(layerNum)] = tf.reshape( self.a['a%d'%(layerNum)], [-1,self.convDataShape[layerNum]] )

        # fc layers
        self.h['h%d' % (layerNum+1)] = \
            net_ops.layer_fullyConn(
                self.a['a%d'%(layerNum)],self.weights['fc_w1'],self.biases['fc_b1'])
        self.a['a%d' % (layerNum+1)] = net_ops.layer_activation(self.h['h%d' % (layerNum+1)], self.activationFunc)
        for layerNum in range( 1 , len( self.fcSize )+1 ):
            absLayerNum = len( self.convNumFilters ) + layerNum + 1
            self.h['h%d' % (absLayerNum)] = \
                net_ops.layer_fullyConn(self.a['a%d'%(absLayerNum-1)], self.weights['fc_w%d'%(layerNum+1)],
                                        self.biases['fc_b%d'%(layerNum+1)])
            if layerNum == len( self.fcSize ):
                self.a['a%d' % (absLayerNum)] = \
                    net_ops.layer_activation(self.h['h%d' % (absLayerNum)], 'linear')
            else:
                self.a['a%d' % (absLayerNum)] = \
                    net_ops.layer_activation(self.h['h%d' % (absLayerNum)], self.activationFunc)

        # output of final layer
        self.y_pred = self.a['a%d' % (absLayerNum)]

        self.numLayers = len(self.a) - 1


    def add_train_op(self,name, balance_classes=True, learning_rate=1e-3, decay_steps=None, decay_rate=None,
                     piecewise_bounds=None, piecewise_values=None, method='Adam', wd_lambda=0.0 ):
        """ Constructs a loss op and training op from a specific loss function and optimiser. User gives the ops a \
            name, and the train op and loss opp are stored in a dictionary (train_ops) under that name.

        Args:
            name (str): Name of the training op (to refer to it later in-case of multiple training ops).
            balance_classes (boolean): Weight the samples during training so that the contribtion to the loss of each \
                            class is balanced by the number of samples the class has (in a given batch).
            learning_rate (float): Controls the degree to which the weights are updated during training.
            decay_steps (int): Epoch frequency at which to decay the learning rate.
            decay_rate (float): Fraction at which to decay the learning rate.
            piecewise_bounds (int list): Epoch step intervals for decaying the learning rate. Alternative to decay steps.
            piecewise_values (float list): Rate at which to decay the learning rate at the piecewise_bounds.
            method (str): Optimisation method.
            wd_lambda (float): Scalar to control weighting of weight decay in loss.

        """
        # construct loss op
        if balance_classes:
            class_weights = net_ops.balance_classes(self.y_target,self.numClasses)
        else:
            class_weights = None
        self.train_ops['%s_loss'%name] = net_ops.loss_function_crossentropy_1D(
            self.y_pred, self.y_target, class_weights=class_weights, num_classes=self.numClasses)

        # weight decay loss contribution
        wdLoss = net_ops.loss_weight_decay(wd_lambda)

        # construct training op
        self.train_ops['%s_train'%name] = \
            net_ops.train_step(self.train_ops['%s_loss'%name]+wdLoss, learning_rate, decay_steps, decay_rate,
                               piecewise_bounds, piecewise_values,method)




    def train(self, dataTrain, dataVal, train_op_name, n_epochs, save_addr, visualiseRateTrain=0, visualiseRateVal=0,
              save_epochs=[1000]):
        """ Calls network_ops function to train a network.

        Args:
            dataTrain (obj): Iterator object for training data.
            dataVal (obj): Iterator object for validation data.
            train_op_name (str): Name of training op created.
            n_epochs (int): Number of loops through dataset to train for.
            save_addr (str): Address of a directory to save checkpoints for desired epochs, or address of saved \
                        checkpoint. If address is for an epoch and contains a previously saved checkpoint, then the \
                        network will start training from there. Otherwise it will be trained from scratch.
            visualiseRateTrain (int): Epoch rate at which to print training loss in console.
            visualiseRateVal (int): Epoch rate at which to print validation loss in console.
            save_epochs (int list): Epochs to save checkpoints at.
        """

        # make sure a checkpoint is saved at n_epochs
        if n_epochs not in save_epochs:
            save_epochs.append(n_epochs)


        net_ops.train( self, dataTrain, dataVal, train_op_name, n_epochs, save_addr, visualiseRateTrain,
                       visualiseRateVal, save_epochs )


    def add_model(self,addr,modelName):
        """ Loads a saved set of model parameters for the network.

        Args:
            addr (str): Address of the directory containing the checkpoint files.
            modelName (str): Name of the model (to refer to it later in-case of multiple models for a given network).
        """

        self.modelsAddrs[modelName] = addr

    def predict_scores( self, modelName, dataSamples, useSoftmax=True  ):
        """ Extract the predicted classification scores of some dataSamples using a trained model.

        Args:
            modelName (str): Name of the model to use (previously added with add_model() ).
            dataSamples (np.array): Shape [numSamples x inputSize].
            useSoftmax (boolean): Pass predicted scores output by network through a softmax function.

        Returns:
            (np.array): Predicted classification scores of dataSamples. Shape [numSamples x numClasses].

        """

        with tf.Session() as sess:

            # load the model
            net_ops.load_model(self.modelsAddrs[modelName], sess)

            # get scores
            if useSoftmax:
                scores = tf.nn.softmax(self.y_pred)
            else:
                scores = self.y_pred
            predScores = sess.run(scores, feed_dict={self.x: dataSamples})

            return predScores


    def predict_labels( self, modelName, dataSamples  ):
        """ Extract the predicted classification labels of some dataSamples using a trained model.

        Args:
            modelName (str): Name of the model to use (previously added with add_model() )
            dataSamples (array): Shape [numSamples x inputSize]

        Returns:
            (np.array): Predicted classification labels of dataSamples. Shape [numSamples].

        """

        with tf.Session() as sess:

            # load the model
            net_ops.load_model(self.modelsAddrs[modelName], sess)

            pred_labels = sess.run(tf.math.argmax(self.y_pred,axis=1), feed_dict={self.x: dataSamples}) + 1

            return pred_labels


    def predict_features( self, modelName, dataSamples, layer ):
        """ Extract the predicted feature values at a particular layer of the network.

        Args:
            modelName (str): Name of the model to use (previously added with add_model() )
            dataSamples (np.array): Shape [numSamples x inputSize]
            layer (int): Layer at which to extract features. Must be between 1 and numLayers inclusive.

        Returns:
            (np.array): Values of neurons at layer. Shape [numSamples x numNeurons] if fully-connected layer and \
                        [numSamples x convDim1 x convDim2] if convolutional layer.

        """

        if (layer > self.numLayers ) | (layer < 1 ):
            raise ValueError('layer must be between 1 and numLayers (%i) inclusive. '
                             'Layer input: %i'% (self.numLayers,layer) )

        with tf.Session() as sess:

            # load the model
            net_ops.load_model(self.modelsAddrs[modelName], sess)

            predFeatures = sess.run(self.a['a%i'%(layer)], feed_dict={self.x: dataSamples})

            return predFeatures














