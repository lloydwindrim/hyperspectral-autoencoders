'''
    Description: high-level deep learning classes for building, training and using unsupervised autoencoders. Uses
    functions from the low-level network_ops module.

    - File name: autoencoder.py
    - Author: Lloyd Windrim
    - Date created: June 2019
    - Python package: deephyp



'''

import tensorflow as tf
from deephyp import network_ops as net_ops


class mlp_1D_network():
    """ Class for setting up a 1-D multi-layer perceptron (mlp) autoencoder network. Layers are all fully-connected \
        (i.e. dense).

    Args:
        configFile (str): Optional way of setting up the network. All other inputs can be ignored (will be overwritten). \
                        Pass the address of the .json config file.
        inputSize (int): Number of dimensions of input data (i.e. number of spectral bands). Value must be input if \
                        not using a config file.
        encoderSize (int list): Number of nodes at each layer of the encoder. List length is number of encoder layers.
        activationFunc (str): Activation function for all layers except the last one. Current options: ['sigmoid', \
                        'relu', 'linear'].
        tiedWeights (binary list or None): Specifies whether or not to tie weights at each layer:
                        - 1: tied weights of specific encoder layer to corresponding decoder weights
                        - 0: do not tie weights of specific layer
                        - None: sets all layers to 0
        weightInitOpt (string): Method of weight initialisation. Current options: ['gaussian', 'truncated_normal', \
                        'xavier', 'xavier_improved'].
        weightStd (float): Used by 'gaussian' and 'truncated_normal' weight initialisation methods.
        skipConnect (boolean): Whether to use skip connections throughout the network.
        activationFuncFinal (str): Activation function for final layer. Current options: ['sigmoid', 'relu', 'linear'].

    Attributes:
        inputSize (int): Number of dimensions of input data (i.e. number of spectral bands).
        activationFunc (str): Activation function for all layers except the last one.
        tiedWeights (binary list): Whether (1) or not (0) the weights of an encoder layer are tied to a decoder layer.
        skipConnect (boolean): Whether the network uses skip connections between corresponding encoder and decoder layers.
        weightInitOpt (string): Method of weight initialisation.
        weightStd (float): Parameter for 'gaussian' and 'truncated_normal' weight initialisation methods.
        activationFuncFinal (str): Activation function for final layer.
        encoderSize (int list): Number of inputs and number of nodes at each layer of the encoder.
        decoderSize (int list): Number of nodes at each layer of the decoder and number of outputs.
        z (tensor): Latent representation of data. Accessible through the *encoder* class function, requiring a trained \
            model.
        y_recon (tensor): Reconstructed output of network. Accessible through the *decoder* and *encoder_decoder* class \
            functions, requiring a trained model.
        train_ops (dict): Dictionary of names of train and loss ops (suffixed with _train and _loss) added to the \
            network using the *add_train_op* class function. The name (without suffix) is passed to the *train* class \
            function to train the network with the referenced train and loss op.
        modelsAddrs (dict): Dictionary of model names added to the network using the *add_model* class function. The \
            names reference models which can be used by the *encoder*, *decoder* and *encoder_decoder* class functions.

    """

    def __init__( self , configFile=None, inputSize=None , encoderSize=[50,30,10] , activationFunc='sigmoid' ,
                  tiedWeights=None , weightInitOpt='truncated_normal' , weightStd=0.1, skipConnect=False,
                  activationFuncFinal='linear'  ):

        self.inputSize = inputSize
        self.activationFunc = activationFunc
        self.tiedWeights = tiedWeights
        self.skipConnect = skipConnect
        self.weightInitOpt = weightInitOpt
        self.weightStd = weightStd
        self.encodersize = encoderSize
        self.activationFuncFinal = activationFuncFinal

        self.net_config = ['inputSize','encodersize','activationFunc','tiedWeights','weightInitOpt','weightStd',
                           'skipConnect','activationFuncFinal']
        # loading config file overwrites input arguments
        if configFile is not None:
            net_ops.load_config(self,configFile)

        if self.inputSize is None:
            raise Exception('value must be given for inputSize (not None)')

        self.encoderSize = [self.inputSize] + self.encodersize
        self.decoderSize = self.encoderSize[::-1]

        self.x = tf.placeholder("float", [None, self.inputSize])
        self.y_target = tf.placeholder("float", [None, self.inputSize])

        self.weights = { }
        self.biases = { }
        self.h = {}
        self.a = {}
        self.train_ops = {}
        self.modelsAddrs = {}

        if self.tiedWeights is None:
            self.tiedWeights = [0]*(len(self.encoderSize)-1)

        # encoder weights
        for layerNum in range( len( self.encoderSize ) - 1 ):
            self.weights['encoder_w%i'%(layerNum+1)] = \
                net_ops.create_variable([self.encoderSize[layerNum],
                                         self.encoderSize[layerNum+1]],self.weightInitOpt, wd=True)

        # decoder weights
        for layerNum in range( len( self.decoderSize ) - 1 ):
            if self.tiedWeights[layerNum] == 0:
                self.weights['decoder_w%i' % (len( self.encoderSize ) + layerNum )] = \
                    net_ops.create_variable([self.decoderSize[layerNum], self.decoderSize[layerNum + 1]],
                                            self.weightInitOpt, wd=True)
            elif self.tiedWeights[layerNum] == 1:
                self.weights['decoder_w%i' % (len(self.encoderSize) + layerNum)] = \
                    tf.transpose( self.weights['encoder_w%i'%(len(self.encoderSize)-1-layerNum)] )
            else:
                raise ValueError('unknown tiedWeights value: %i. '
                                 'Must be 0 or 1 for each layer (or None).' % tiedWeights[layerNum])



        # encoder biases
        for layerNum in range( len( self.encoderSize ) - 1 ):
            self.biases['encoder_b%i'%(layerNum+1)] = \
                net_ops.create_variable([self.encoderSize[layerNum+1]] , self.weightInitOpt, wd=True)

        # decoder biases
        for layerNum in range( len( self.decoderSize ) - 1 ):
            self.biases['decoder_b%i' % (len( self.encoderSize ) + layerNum )] = \
                net_ops.create_variable([self.decoderSize[layerNum + 1]], self.weightInitOpt, wd=True)

        # build network using encoder, decoder and x placeholder as input

        # build encoder
        self.a['a0'] = self.x
        for layerNum in range( 1 , len( self.encoderSize ) ):
            self.h['h%d' % (layerNum)] = \
                net_ops.layer_fullyConn(self.a['a%d'%(layerNum-1)], self.weights['encoder_w%d'%(layerNum)],
                                        self.biases['encoder_b%d'%(layerNum)])
            self.a['a%d' % (layerNum)] = net_ops.layer_activation(self.h['h%d' % (layerNum)], self.activationFunc)

        # latent representation
        self.z = self.a['a%d' % (layerNum)]

        # build decoder
        for layerNum in range( 1 , len( self.decoderSize ) ):
            absLayerNum = len(self.encoderSize) + layerNum - 1
            self.h['h%d' % (absLayerNum)] = \
                net_ops.layer_fullyConn(self.a['a%d'%(absLayerNum-1)], self.weights['decoder_w%d'%(absLayerNum)],
                                        self.biases['decoder_b%d'%(absLayerNum)])
            if layerNum < len( self.decoderSize )-1:
                if self.skipConnect:
                    self.h['h%d' % (absLayerNum)] += self.h['h%d' % (len(self.decoderSize) - layerNum - 1)]
                self.a['a%d' % (absLayerNum)] = \
                    net_ops.layer_activation(self.h['h%d' % (absLayerNum)], self.activationFunc)
            else:
                if self.skipConnect:
                    self.h['h%d' % (absLayerNum)] += self.a['a0']
                self.a['a%d' % (absLayerNum)] = \
                    net_ops.layer_activation(self.h['h%d' % (absLayerNum)], self.activationFuncFinal)

        # output of final layer
        self.y_recon = self.a['a%d' % (absLayerNum)]



    def add_train_op(self,name,lossFunc='CSA',learning_rate=1e-3, decay_steps=None, decay_rate=None,
                     piecewise_bounds=None, piecewise_values=None, method='Adam', wd_lambda=0.0 ):
        """ Constructs a loss op and training op from a specific loss function and optimiser. User gives the ops a \
            name, and the train op and loss opp are stored in a dictionary (train_ops) under that name.

        Args:
            name (str): Name of the training op (to refer to it later in-case of multiple training ops).
            lossFunc (str): Reconstruction loss function.
            learning_rate (float): Controls the degree to which the weights are updated during training.
            decay_steps (int): Epoch frequency at which to decay the learning rate.
            decay_rate (float): Fraction at which to decay the learning rate.
            piecewise_bounds (int list): Epoch step intervals for decaying the learning rate. Alternative to decay steps.
            piecewise_values (float list): Rate at which to decay the learning rate at the piecewise_bounds.
            method (str): Optimisation method.
            wd_lambda (float): Scalar to control weighting of weight decay in loss.

        """
        # construct loss op
        self.train_ops['%s_loss'%name] = net_ops.loss_function_reconstruction_1D(self.y_recon, self.y_target, func=lossFunc)

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

    def encoder( self, modelName, dataSamples  ):
        """ Extract the latent variable of some dataSamples using a trained model.

        Args:
            modelName (str): Name of the model to use (previously added with add_model() ).
            dataSample (np.array): Shape [numSamples x inputSize].

        Returns:
            (np.array): Latent representation z of dataSamples. Shape [numSamples x arbitrary].

        """

        with tf.Session() as sess:

            # load the model
            net_ops.load_model(self.modelsAddrs[modelName], sess)

            # get latent values
            dataZ = sess.run(self.z, feed_dict={self.x: dataSamples})

            return dataZ



    def decoder( self, modelName, dataZ  ):
        """ Extract the reconstruction of some dataSamples from their latent representation encoding  using a trained \
            model.

        Args:
            modelName (str): Name of the model to use (previously added with add_model() ).
            dataZ (np.array): Latent representation of data samples to reconstruct using the network. Shape \
                    [numSamples x arbitrary].

        Returns:
            (np.array): Reconstructed data (y_recon attribute). Shape [numSamples x arbitrary].

        """

        with tf.Session() as sess:

            # load the model
            net_ops.load_model(self.modelsAddrs[modelName], sess)

            # get reconstruction
            dataY_recon = sess.run(self.y_recon, feed_dict={self.z: dataZ})

            return dataY_recon


    def encoder_decoder( self, modelName, dataSamples  ):
        """ Extract the reconstruction of some dataSamples using a trained model.

        Args:
            modelName (str): Name of the model to use (previously added with add_model() ).
            dataSample (np.array): Data samples to reconstruct using the network. Shape [numSamples x inputSize].

        Returns:
            (np.array): Reconstructed data (y_recon attribute). Shape [numSamples x arbitrary].

        """

        with tf.Session() as sess:

            # load the model
            net_ops.load_model(self.modelsAddrs[modelName], sess)

            # get reconstruction
            dataY_recon = sess.run(self.y_recon, feed_dict={self.x: dataSamples})

            return dataY_recon





class cnn_1D_network():
    """ Class for setting up a 1-D convolutional autoencoder network. Builds a network with an encoder containing  \
        convolutional layers followed by a single fully-connected layer to map from the final convolutional layer in \
        the encoder to the latent layer. The decoder contains a single fully-connected layer and then several \
        deconvolutional layers which reconstruct the spectra in the output.


    Args:
        configFile (str): Optional way of setting up the network. All other inputs can be ignored (will be overwritten). \
                        Pass the address of the .json config file.
        inputSize (int): Number of dimensions of input data (i.e. number of spectral bands). Value must be input if not \
                        using a config file.
        zDim (int): Dimensionality of latent vector.
        encoderNumFilters (int list): Number of filters at each layer of the encoder. List length is number of \
                        convolutional encoder layers. Note that there is a single mlp layer after the last \
                        convolutional layer.
        encoderFilterSize (int list): Size of filter at each layer of the encoder. List length is number of encoder layers.
        activationFunc (str): Activation function for all layers except the last one. Current options: ['sigmoid', \
                        'relu', 'linear'].
        tiedWeights (binary list or None): Specifies whether or not to tie weights at each layer:
                    - 1: tied weights of specific encoder layer to corresponding decoder weights
                    - 0: do not tie weights of specific layer
                    - None: sets all layers to 0
        weightInitOpt (string): Method of weight initialisation. Current options: ['gaussian', 'truncated_normal', \
                    'xavier', 'xavier_improved'].
        weightStd (float): Used by 'gaussian' and 'truncated_normal' weight initialisation methods.
        skipConnect (boolean): Whether to use skip connections throughout the network.
        padding (str): Type of padding used. Current options: ['VALID', 'SAME'].
        encoderStride (int list): Stride at each convolutional encoder layer.
        activationFuncFinal (str): Activation function for final layer. Current options: ['sigmoid', 'relu', 'linear'].


    Attributes:
        inputSize (int): Number of dimensions of input data (i.e. number of spectral bands).
        activationFunc (str): Activation function for all layers except the last one.
        tiedWeights (binary list): Whether (1) or not (0) the weights of an encoder layer are tied to a decoder layer.
        skipConnect (boolean): Whether the network uses skip connections between corresponding encoder and decoder layers.
        weightInitOpt (string): Method of weight initialisation.
        weightStd (float): Parameter for 'gaussian' and 'truncated_normal' weight initialisation methods.
        activationFuncFinal (str): Activation function for final layer.
        encoderNumFilters (int list): Number of filters at each layer of the encoder. List length is number of \
                        convolutional encoder layers. Note that there is a single mlp layer after the last \
                        convolutional layer.
        encoderFilterSize (int list): Size of filter at each layer of the encoder. List length is number of encoder layers.
        encoderStride (int list): Stride at each convolutional encoder layer.
        decoderNumFilters (int list):
        decoderFilterSize (int list):
        decoderStride (int list):
        zDim (int): Dimensionality of latent vector.
        padding (str): Type of padding used. Current options: ['VALID', 'SAME'].
        z (tensor): Latent representation of data. Accessible through the *encoder* class function, requiring a trained \
            model.
        y_recon (tensor): Reconstructed output of network. Accessible through the *decoder* and *encoder_decoder* class \
            functions, requiring a trained model.
        train_ops (dict): Dictionary of names of train and loss ops (suffixed with _train and _loss) added to the \
            network using the *add_train_op* class function. The name (without suffix) is passed to the *train* class \
            function to train the network with the referenced train and loss op.
        modelsAddrs (dict): Dictionary of model names added to the network using the *add_model* class function. The \
            names reference models which can be used by the *encoder*, *decoder* and *encoder_decoder* class functions.



    """

    def __init__( self , configFile=None, inputSize=None , zDim=5, encoderNumFilters=[10,10,10] ,
                  encoderFilterSize=[20,10,10], activationFunc='sigmoid', tiedWeights=None,
                  weightInitOpt='truncated_normal', weightStd=0.1, skipConnect=False, padding='VALID',
                  encoderStride=[1,1,1], activationFuncFinal='linear' ):


        self.inputSize = inputSize
        self.tiedWeights = tiedWeights
        self.skipConnect = skipConnect
        self.weightInitOpt = weightInitOpt
        self.weightStd = weightStd
        self.zDim = zDim
        self.padding = padding
        self.activationFunc = activationFunc
        self.encoderStride = encoderStride
        self.encoderNumfilters = encoderNumFilters
        self.encoderFiltersize = encoderFilterSize
        self.activationFuncFinal = activationFuncFinal

        self.net_config = ['inputSize','zDim','encoderNumfilters','encoderFiltersize','activationFunc','tiedWeights',
                           'weightInitOpt','weightStd','skipConnect','padding','encoderStride','activationFuncFinal']
        # loading config file overwrites input arguments
        if configFile is not None:
            net_ops.load_config(self,configFile)

        if self.inputSize is None:
            raise Exception('value must be given for inputSize (not None)')

        if not (len(self.encoderFiltersize) == len(self.encoderNumfilters) == len(self.encoderStride)):
            raise Exception('the length of encoderNumfilters, encoderFilterSize and encoderStride must be equal.')


        self.encoderNumFilters = [1] + self.encoderNumfilters
        self.decoderNumFilters = self.encoderNumFilters[::-1]
        self.encoderFilterSize = self.encoderFiltersize
        self.decoderFilterSize = self.encoderFiltersize[::-1]
        self.decoderStride = encoderStride[::-1]


        self.x = tf.placeholder("float", [None, self.inputSize])
        self.y_target = tf.placeholder("float", [None, self.inputSize])

        self.weights = { }
        self.biases = { }
        self.h = {}
        self.a = {}
        self.train_ops = {}
        self.modelsAddrs = {}

        if self.tiedWeights is None:
            self.tiedWeights = [0]*(len(self.encoderNumFilters)-1)

        # pre-compute shape of data after each layer
        self.encoderDataShape = [self.inputSize]
        for layerNum in range( len( self.encoderNumFilters ) - 1 ):
            self.encoderDataShape.append( net_ops.conv_output_shape(
                self.encoderDataShape[layerNum],self.encoderFilterSize[layerNum],self.padding,self.encoderStride[layerNum]) )
        self.encoderDataShape.append(self.zDim)
        self.encoderDataShape[layerNum + 1] = self.encoderDataShape[-2] * self.encoderNumFilters[layerNum + 1]
        self.decoderDataShape = self.encoderDataShape[::-1]

        #--

        # encoder weights
        for layerNum in range( len( self.encoderNumFilters ) - 1 ):
            self.weights['encoder_w%i'%(layerNum+1)] = \
                net_ops.create_variable([self.encoderFilterSize[layerNum], self.encoderNumFilters[layerNum],
                                         self.encoderNumFilters[layerNum+1]],weightInitOpt, wd=True)
        self.weights['encoder_w%i' % (layerNum + 2)] = net_ops.create_variable(
            [self.encoderDataShape[layerNum+1], self.zDim],self.weightInitOpt, wd=True)

        # decoder weights
        self.weights['decoder_w%i' % (layerNum + 3)] = net_ops.create_variable(
            [self.zDim,self.decoderDataShape[1]], self.weightInitOpt, wd=True)
        for layerNum in range( len( self.decoderNumFilters ) - 1  ):
            if self.tiedWeights[layerNum] == 0:
                self.weights['decoder_w%i' % (len( self.encoderDataShape ) + layerNum + 1)] = \
                    net_ops.create_variable([self.decoderFilterSize[layerNum], self.decoderNumFilters[layerNum+1],
                                             self.decoderNumFilters[layerNum]], self.weightInitOpt, wd=True)
            elif self.tiedWeights[layerNum] == 1:
                self.weights['decoder_w%i' % (len( self.encoderNumFilters )+layerNum+2)] = \
                    self.weights['encoder_w%i' % (len(self.encoderNumFilters)-1 - layerNum) ]
            else:
                raise ValueError('unknown tiedWeights value: %i. '
                                 'Must be 0 or 1 for each layer (or None).' % tiedWeights[layerNum])


        # encoder biases
        for layerNum in range( len( self.encoderNumFilters ) - 1 ):
            self.biases['encoder_b%i'%(layerNum+1)] = \
                net_ops.create_variable([self.encoderNumFilters[layerNum+1]] , self.weightInitOpt, wd=True)
        self.biases['encoder_b%i'%(layerNum+2)] = net_ops.create_variable([self.zDim] , self.weightInitOpt, wd=True)

        # decoder biases
        self.biases['decoder_b%i' % (layerNum + 3)] = \
            net_ops.create_variable([self.decoderDataShape[1]], self.weightInitOpt, wd=True)
        for layerNum in range( len( self.decoderNumFilters ) - 1  ):
            self.biases['decoder_b%i' % (len( self.encoderDataShape ) + layerNum + 1)] = \
                net_ops.create_variable([self.decoderNumFilters[layerNum+1]], self.weightInitOpt, wd=True)

        # build network using encoder, decoder and x placeholder as input

        # build encoder
        self.a['a0'] = tf.expand_dims(self.x,axis=2)   # expand to shape None x inputSize x 1
        for layerNum in range( 1 , len( self.encoderNumFilters ) ):
            self.h['h%d' % (layerNum)] = \
                net_ops.layer_conv1d(self.a['a%d'%(layerNum-1)], self.weights['encoder_w%d'%(layerNum)],
                                     self.biases['encoder_b%d'%(layerNum)],padding=self.padding,stride=self.encoderStride[layerNum-1])
            self.a['a%d' % (layerNum)] = net_ops.layer_activation(self.h['h%d' % (layerNum)], self.activationFunc)
        self.a['a%d'%(layerNum)] = tf.reshape( self.a['a%d'%(layerNum)], [-1,self.encoderDataShape[layerNum]] )
        self.h['h%d' % (layerNum+1)] = \
            net_ops.layer_fullyConn(
                self.a['a%d'%(layerNum)],self.weights['encoder_w%d'%(layerNum+1)],self.biases['encoder_b%d'%(layerNum+1)])
        self.a['a%d' % (layerNum+1)] = net_ops.layer_activation(self.h['h%d' % (layerNum+1)], self.activationFunc)


        # latent representation
        self.z = self.a['a%d' % (layerNum+1)] # collapse a dim

        # build decoder
        self.h['h%d' % (layerNum+2)] = \
            net_ops.layer_fullyConn(self.a['a%d' % (layerNum+1)],
                                    self.weights['decoder_w%d' % (layerNum+2)],self.biases['decoder_b%d' % (layerNum+2)])
        if skipConnect:
            self.h['h%d' % (layerNum+2)] += tf.reshape(
                self.h['h%d' % (len( self.decoderNumFilters ) - 1)] , [-1,self.encoderDataShape[layerNum]] )
        self.a['a%d' % (layerNum+2)] = net_ops.layer_activation(self.h['h%d' % (layerNum+2)], self.activationFunc)
        self.a['a%d' % (layerNum + 2)] = tf.reshape(
            self.a['a%d' % (layerNum + 2)], [-1,int(self.decoderDataShape[1]/self.encoderNumFilters[-1]),self.encoderNumFilters[-1]] )
        for layerNum in range( 1 , len( self.decoderNumFilters ) ):
            absLayerNum = len( self.encoderDataShape ) + layerNum
            outputShape = [tf.shape(self.a['a%d' % (absLayerNum-1)] )[0],
                           self.decoderDataShape[layerNum+1],self.decoderNumFilters[layerNum]]
            self.h['h%d' % (absLayerNum)] = \
                net_ops.layer_deconv1d(self.a['a%d'%(absLayerNum-1)], self.weights['decoder_w%d'%(absLayerNum)],
                                       self.biases['decoder_b%d'%(absLayerNum)],
                                       outputShape, padding=self.padding, stride=self.decoderStride[layerNum-1])
            if layerNum < len( self.decoderNumFilters )-1:
                if self.skipConnect:
                    self.h['h%d' % (absLayerNum)] += self.h['h%d' % (len( self.decoderNumFilters ) - layerNum - 1)]
                self.a['a%d' % (absLayerNum)] = net_ops.layer_activation(self.h['h%d' % (absLayerNum)], self.activationFunc)
            else:
                if self.skipConnect:
                    self.h['h%d' % (absLayerNum)] += self.a['a0']
                self.a['a%d' % (absLayerNum)] = net_ops.layer_activation(self.h['h%d' % (absLayerNum)], self.activationFuncFinal)

        # output of final layer
        self.y_recon = tf.squeeze( self.a['a%d' % (absLayerNum)] , axis=2)



    def add_train_op(self,name,lossFunc='SSE',learning_rate=1e-3, decay_steps=None, decay_rate=None,
                     piecewise_bounds=None, piecewise_values=None, method='Adam', wd_lambda=0.0 ):
        """ Constructs a loss op and training op from a specific loss function and optimiser. User gives the ops a name, \
            and the train op and loss opp are stored in a dictionary (train_ops) under that name.

        Args:
            name (str): Name of the training op (to refer to it later in-case of multiple training ops).
            lossFunc (str): Reconstruction loss function.
            learning_rate (float): Controls the degree to which the weights are updated during training.
            decay_steps (int): Epoch frequency at which to decay the learning rate.
            decay_rate (float): Fraction at which to decay the learning rate.
            piecewise_bounds (int list): Epoch step intervals for decaying the learning rate. Alternative to decay steps.
            piecewise_values (float list): Rate at which to decay the learning rate at the piecewise_bounds.
            method (str): Optimisation method.
            wd_lambda (float): Scalar to control weighting of weight decay in loss.

        """
        # construct loss op
        self.train_ops['%s_loss'%name] = net_ops.loss_function_reconstruction_1D(self.y_recon, self.y_target, func=lossFunc)

        # weight decay loss contribution
        wdLoss = net_ops.loss_weight_decay(wd_lambda)

        # construct training op
        self.train_ops['%s_train'%name] = \
            net_ops.train_step(self.train_ops['%s_loss'%name]+wdLoss, learning_rate, decay_steps, decay_rate, piecewise_bounds, piecewise_values,method)




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

        net_ops.train( self, dataTrain, dataVal, train_op_name, n_epochs, save_addr, visualiseRateTrain, visualiseRateVal, save_epochs )


    def add_model(self,addr,modelName):
        """ Loads a saved set of model parameters for the network.

        Args:
            addr (str): Address of the directory containing the checkpoint files.
            modelName (str): Name of the model (to refer to it later in-case of multiple models for a given network).
        """

        self.modelsAddrs[modelName] = addr


    def encoder( self, modelName, dataSamples  ):
        """ Extract the latent variable of some dataSamples using a trained model.

        Args:
            modelName (str): Name of the model to use (previously added with add_model() ).
            dataSample (np.array): Shape [numSamples x inputSize].

        Returns:
            (np.array): Latent representation z of dataSamples. Shape [numSamples x arbitrary].

        """

        with tf.Session() as sess:

            # load the model
            net_ops.load_model(self.modelsAddrs[modelName], sess)

            # get latent values
            dataZ = sess.run(self.z, feed_dict={self.x: dataSamples})

            return dataZ



    def decoder( self, modelName, dataZ  ):
        """ Extract the reconstruction of some dataSamples from their latent representation encoding  using a trained \
            model.

        Args:
            modelName (str): Name of the model to use (previously added with add_model() ).
            dataZ (np.array): Latent representation of data samples to reconstruct using the network. Shape \
                    [numSamples x arbitrary].

        Returns:
            (np.array): Reconstructed data (y_recon attribute). Shape [numSamples x arbitrary].

        """

        with tf.Session() as sess:

            # load the model
            net_ops.load_model(self.modelsAddrs[modelName], sess)

            # get reconstruction
            dataY_recon = sess.run(self.y_recon, feed_dict={self.z: dataZ})

            return dataY_recon


    def encoder_decoder( self, modelName, dataSamples  ):
        """ Extract the reconstruction of some dataSamples using a trained model.

        Args:
            modelName (str): Name of the model to use (previously added with add_model() ).
            dataSample (np.array): Data samples to reconstruct using the network. Shape [numSamples x inputSize].

        Returns:
            (np.array): Reconstructed data (y_recon attribute). Shape [numSamples x arbitrary].

        """

        with tf.Session() as sess:

            # load the model
            net_ops.load_model(self.modelsAddrs[modelName], sess)

            # get reconstruction
            dataY_recon = sess.run(self.y_recon, feed_dict={self.x: dataSamples})

            return dataY_recon







