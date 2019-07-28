import tensorflow as tf
import network_ops as net_ops




class mlp_1D_network():

    def __init__( self , inputSize , encoderSize=[50,30,10] , activationFunc='sigmoid' ,
                  tiedWeights=None , weightInitOpt='truncated_normal' , weightStd=0.1, skipConnect=False  ):

        """ Class for setting up a 1-D multi-layer perceptron autoencoder network.
        - input:
            inputSize: (int) Number of dimensions of input data (i.e. number of spectral bands)
            encoderSize: (int list) Number of nodes at each layer of the encoder. List length is number of encoder layers.
            activationFunc: (string) [sigmoid, ReLU, tanh]
            tiedWeights: (binary list or None) 1 - tied weights of specific encoder layer to corresponding decoder weights.
                                        0 - do not not weights of specific layer
                                        None - sets all layers to 0
            weightInitOpt: (string) Method of weight initialisation. [gaussian, truncated_normal, xavier, xavier_improved]
            weightStd: (float) Used by 'gaussian' and 'truncated_normal' weight initialisation methods
        """

        self.numBands = inputSize
        self.encoderSize = [inputSize] + encoderSize
        self.decoderSize = self.encoderSize[::-1]
        self.activationFunc = activationFunc
        self.tiedWeights = tiedWeights
        self.weightInitOpt = weightInitOpt
        self.weightStd = weightStd

        self.x = tf.placeholder("float", [None, inputSize])
        self.y_target = tf.placeholder("float", [None, inputSize])

        self.weights = { }
        self.biases = { }
        self.h = {}
        self.a = {}
        self.train_ops = {}
        self.modelsAddrs = {}

        if tiedWeights is None:
            tiedWeights = [0]*len(encoderSize)

        # encoder weights
        for layerNum in range( len( self.encoderSize ) - 1 ):
            self.weights['encoder_w%i'%(layerNum+1)] = \
                net_ops.create_variable([self.encoderSize[layerNum], self.encoderSize[layerNum+1]],weightInitOpt, wd=True)

        # decoder weights
        for layerNum in range( len( self.decoderSize ) - 1 ):
            if tiedWeights[layerNum] == 0:
                self.weights['decoder_w%i' % (len( self.encoderSize ) + layerNum )] = \
                    net_ops.create_variable([self.decoderSize[layerNum], self.decoderSize[layerNum + 1]], weightInitOpt, wd=True)
            elif tiedWeights[layerNum] == 1:
                self.weights['decoder_w%i' % (len(self.encoderSize) + layerNum)] = \
                    tf.transpose( self.weights['encoder_w%i'%(len(self.encoderSize)-1-layerNum)] )
            else:
                pass
            # catch error


        # encoder biases
        for layerNum in range( len( self.encoderSize ) - 1 ):
            self.biases['encoder_b%i'%(layerNum+1)] = \
                net_ops.create_variable([self.encoderSize[layerNum+1]] , weightInitOpt, wd=True)

        # decoder biases
        for layerNum in range( len( self.decoderSize ) - 1 ):
            self.biases['decoder_b%i' % (len( self.encoderSize ) + layerNum )] = \
                net_ops.create_variable([self.decoderSize[layerNum + 1]], weightInitOpt, wd=True)

        # build network using encoder, decoder and x placeholder as input

        # build encoder
        self.a['a0'] = self.x
        for layerNum in range( 1 , len( self.encoderSize ) ):
            self.h['h%d' % (layerNum)] = \
                net_ops.layer_fullyConn(self.a['a%d'%(layerNum-1)], self.weights['encoder_w%d'%(layerNum)], self.biases['encoder_b%d'%(layerNum)])
            self.a['a%d' % (layerNum)] = net_ops.layer_activation(self.h['h%d' % (layerNum)], activationFunc)

        # latent representation
        self.z = self.a['a%d' % (layerNum)]

        # build decoder
        for layerNum in range( 1 , len( self.decoderSize ) ):
            absLayerNum = len(self.encoderSize) + layerNum - 1
            self.h['h%d' % (absLayerNum)] = \
                net_ops.layer_fullyConn(self.a['a%d'%(absLayerNum-1)], self.weights['decoder_w%d'%(absLayerNum)], self.biases['decoder_b%d'%(absLayerNum)])
            if layerNum < len( self.decoderSize )-1:
                if skipConnect:
                    self.h['h%d' % (absLayerNum)] += self.h['h%d' % (len(self.decoderSize) - layerNum - 1)]
                self.a['a%d' % (absLayerNum)] = net_ops.layer_activation(self.h['h%d' % (absLayerNum)], activationFunc)
            else:
                if skipConnect:
                    self.h['h%d' % (absLayerNum)] += self.a['a0']
                self.a['a%d' % (absLayerNum)] = net_ops.layer_activation(self.h['h%d' % (absLayerNum)], 'linear')

        # output of final layer
        self.y_recon = self.a['a%d' % (absLayerNum)]


    def add_train_op(self,name,lossFunc='SSE',learning_rate=1e-3, decay_steps=None, decay_rate=None, piecewise_bounds=None, piecewise_values=None,
             method='Adam', wd_lambda=0.0 ):
        """ Constructs a loss op and training op from a specific loss function and optimiser. User gives the train op a name, and the train op
            and loss opp are stored in a dictionary under that name
        - input:
            name: (str) Name of the training op (to refer to it later in-case of multiple training ops).
            lossFunc: (str) Reconstruction loss function
            learning rate: (float)
            decay_steps: (int) epoch frequency at which to decay the learning rate.
            decay_rate: (float) fraction at which to decay the learning rate.
            piecewise_bounds: (int list) epoch step intervals for decaying the learning rate. Alternative to decay steps.
            piecewise_values: (float list) rate at which to decay the learning rate at the piecewise_bounds.
            method: (str) optimisation method.
            wd_lambda: (float) scalar to control weighting of weight decay in loss.

        """
        # construct loss op
        self.train_ops['%s_loss'%name] = net_ops.loss_function_reconstruction_1D(self.y_recon, self.y_target, func=lossFunc)

        # weight decay loss contribution
        wdLoss = net_ops.loss_weight_decay(wd_lambda)

        # construct training op
        self.train_ops['%s_train'%name] = \
            net_ops.train_step(self.train_ops['%s_loss'%name]+wdLoss, learning_rate, decay_steps, decay_rate, piecewise_bounds, piecewise_values,method)




    def train(self, dataTrain, dataVal, train_op_name, n_epochs, save_addr, visualiseRateTrain=0, visualiseRateVal=0, save_epochs=[1000]):
        """ Calls network_ops function to train a network.
        - input:
            dataTrain: (obj) iterator object for training data.
            dataVal: (obj) iterator object for validation data.
            train_op_name: (string) name of training op created.
            n_epochs: (int) number of loops through dataset to train for.
            save_addr: (str) address of a directory to save checkpoints for desired epochs.
            visualiseRateTrain: (int) epoch rate at which to print training loss in console
            visualiseRateVal: (int) epoch rate at which to print validation loss in console
            save_epochs: (int list) epochs to save checkpoints at.
        """

        net_ops.train( self, dataTrain, dataVal, train_op_name, n_epochs, save_addr, visualiseRateTrain, visualiseRateVal, save_epochs )


    def add_model(self,addr,name):

        self.modelsAddrs[name] = addr

    def encoder( self, modelName, dataSamples  ):
        """ Extract the latent variable of some dataSamples using a trained model
        - input:
            modelName: (str) Name of the model to use (previously added with add_model() )
            dataSample: (array) Shape [numSamples x numBands]
        - output:
            dataZ: (array) Shape [numSamples x arbitrary]

        """

        with tf.Session() as sess:

            # load the model
            net_ops.load_model(self.modelsAddrs[modelName], sess)

            # get latent values
            dataZ = sess.run(self.z, feed_dict={self.x: dataSamples})

            return dataZ



    def decoder( self, modelName, dataZ  ):
        """ Extract the reconstruction of some dataSamples (with latent representation) using a trained model
        - input:
            modelName: (str) Name of the model to use (previously added with add_model() )
            dataZ: (array) Latent representation of data samples. Shape [numSamples x arbitrary]
        - output:
            dataY_recon: (array) Reconstructed data. Shape [numSamples x arbitrary]

        """

        with tf.Session() as sess:

            # load the model
            net_ops.load_model(self.modelsAddrs[modelName], sess)

            # get reconstruction
            dataY_recon = sess.run(self.y_recon, feed_dict={self.z: dataZ})

            return dataY_recon


    def encoder_decoder( self, modelName, dataSamples  ):
        """ Extract the reconstruction of some dataSamples using a trained model
        - input:
            modelName: (str) Name of the model to use (previously added with add_model() )
            dataSample: (array) Shape [numSamples x numBands]
        - output:
            dataY_recon: (array) Reconstructed data. Shape [numSamples x arbitrary]

        """

        with tf.Session() as sess:

            # load the model
            net_ops.load_model(self.modelsAddrs[modelName], sess)

            # get reconstruction
            dataY_recon = sess.run(self.y_recon, feed_dict={self.x: dataSamples})

            return dataY_recon





class cnn_1D_network():

    def __init__( self , inputSize , zDim =5, encoderNumFilters=[10,10,10] , encoderFilterSize=[20,10,10], activationFunc='sigmoid' ,
                  tiedWeights=None , weightInitOpt='truncated_normal' , weightStd=0.1, skipConnect=False, padding='VALID', stride=[1,1,1]  ):

        """ Class for setting up a 1-D multi-layer perceptron autoencoder network.
        - input:
            inputSize: (int) Number of dimensions of input data (i.e. number of spectral bands)
            encoderNumFilters: (int list) Number of filters at each layer of the encoder. List length is number of encoder layers - 1.
                                Note that the number of filters in the final layer of the encoder will always be 1.
            encoderFilterSize: (int list) Size of filter at each layer of the encoder. List length is number of encoder layers.
            activationFunc: (string) [sigmoid, ReLU, tanh]
            tiedWeights: (binary list or None) 1 - tied weights of specific encoder layer to corresponding decoder weights.
                                        0 - do not not weights of specific layer
                                        None - sets all layers to 0
            weightInitOpt: (string) Method of weight initialisation. [gaussian, truncated_normal, xavier, xavier_improved]
            weightStd: (float) Used by 'gaussian' and 'truncated_normal' weight initialisation methods
            stride: (int list)
            padding: (string)
        """

        self.numBands = inputSize
        self.encoderNumFilters = [1] + encoderNumFilters
        self.decoderNumFilters = self.encoderNumFilters[::-1]
        self.encoderFilterSize = encoderFilterSize
        self.decoderFilterSize = encoderFilterSize[::-1]
        self.activationFunc = activationFunc
        self.tiedWeights = tiedWeights
        self.weightInitOpt = weightInitOpt
        self.weightStd = weightStd
        self.zDim = zDim
        self.encoderStride = stride
        self.decoderStride = stride[::-1]

        self.x = tf.placeholder("float", [None, inputSize])
        self.y_target = tf.placeholder("float", [None, inputSize])

        self.weights = { }
        self.biases = { }
        self.h = {}
        self.a = {}
        self.train_ops = {}
        self.modelsAddrs = {}

        if tiedWeights is None:
            tiedWeights = [0]*(len(self.encoderNumFilters)-1)

        # pre-compute shape of data after each layer
        self.encoderDataShape = [inputSize]
        for layerNum in range( len( encoderNumFilters ) ):
            self.encoderDataShape.append( net_ops.conv_output_shape(self.encoderDataShape[layerNum],encoderFilterSize[layerNum],padding,stride[layerNum]) )
        self.encoderDataShape.append(self.zDim)
        self.encoderDataShape[layerNum + 1] = self.encoderDataShape[-2] * self.encoderNumFilters[layerNum + 1]
        self.decoderDataShape = self.encoderDataShape[::-1]

        #--

        # encoder weights
        for layerNum in range( len( encoderNumFilters ) ):
            self.weights['encoder_w%i'%(layerNum+1)] = \
                net_ops.create_variable([self.encoderFilterSize[layerNum], self.encoderNumFilters[layerNum], self.encoderNumFilters[layerNum+1]],weightInitOpt, wd=True)
        self.weights['encoder_w%i' % (layerNum + 2)] = net_ops.create_variable([self.encoderDataShape[layerNum+1], self.zDim],weightInitOpt, wd=True)

        # decoder weights
        if tiedWeights[layerNum] == 0:
            self.weights['decoder_w%i' % (layerNum + 3)] = net_ops.create_variable(
                [self.zDim,self.decoderDataShape[1]], weightInitOpt, wd=True)
            for layerNum in range( len( self.decoderNumFilters ) - 1  ):
                self.weights['decoder_w%i' % (len( self.encoderDataShape ) + layerNum + 1)] = \
                    net_ops.create_variable([self.decoderFilterSize[layerNum], self.decoderNumFilters[layerNum+1], self.decoderNumFilters[layerNum]], weightInitOpt, wd=True)
        elif tiedWeights[layerNum] == 1:
            pass
            # self.weights['decoder_w%i' % (layerNum + 3)] = net_ops.create_variable(
            #     [self.zDim,self.decoderDataShape[1]], weightInitOpt, wd=True)
            # for layerNum in range(len(self.decoderNumFilters) - 1):
            #     self.weights['decoder_w%i' % (len( self.decoderNumFilters ) + layerNum )] = \
            #         tf.transpose( self.weights['encoder_w%i'%(len(self.encoderSize)-1-layerNum)] )
        else:
            pass
            # catch error


        # encoder biases
        for layerNum in range( len( encoderNumFilters ) ):
            self.biases['encoder_b%i'%(layerNum+1)] = \
                net_ops.create_variable([self.encoderNumFilters[layerNum+1]] , weightInitOpt, wd=True)
        self.biases['encoder_b%i'%(layerNum+2)] = net_ops.create_variable([self.zDim] , weightInitOpt, wd=True)

        # decoder biases
        self.biases['decoder_b%i' % (layerNum + 3)] = net_ops.create_variable([self.decoderDataShape[1]], weightInitOpt, wd=True)
        for layerNum in range( len( self.decoderNumFilters ) - 1  ):
            self.biases['decoder_b%i' % (len( self.encoderDataShape ) + layerNum + 1)] = \
                net_ops.create_variable([self.decoderNumFilters[layerNum+1]], weightInitOpt, wd=True)

        # build network using encoder, decoder and x placeholder as input

        # build encoder
        self.a['a0'] = tf.expand_dims(self.x,axis=2)   # expand to shape None x numBands x 1
        for layerNum in range( 1 , len( self.encoderNumFilters ) ):
            self.h['h%d' % (layerNum)] = \
                net_ops.layer_conv1d(self.a['a%d'%(layerNum-1)], self.weights['encoder_w%d'%(layerNum)],
                                     self.biases['encoder_b%d'%(layerNum)],padding=padding,stride=stride[layerNum-1])
            self.a['a%d' % (layerNum)] = net_ops.layer_activation(self.h['h%d' % (layerNum)], activationFunc)
        self.a['a%d'%(layerNum)] = tf.reshape( self.a['a%d'%(layerNum)], [-1,self.encoderDataShape[layerNum]] )
        self.h['h%d' % (layerNum+1)] = \
            net_ops.layer_fullyConn(self.a['a%d'%(layerNum)],self.weights['encoder_w%d'%(layerNum+1)],self.biases['encoder_b%d'%(layerNum+1)])
        self.a['a%d' % (layerNum+1)] = net_ops.layer_activation(self.h['h%d' % (layerNum+1)], activationFunc)


        # latent representation
        self.z = self.a['a%d' % (layerNum+1)] # collapse a dim

        # build decoder
        self.h['h%d' % (layerNum+2)] = \
            net_ops.layer_fullyConn(self.a['a%d' % (layerNum+1)], self.weights['decoder_w%d' % (layerNum+2)],self.biases['decoder_b%d' % (layerNum+2)])
        self.a['a%d' % (layerNum+2)] = net_ops.layer_activation(self.h['h%d' % (layerNum+2)], activationFunc)
        self.a['a%d' % (layerNum + 2)] = tf.reshape( self.a['a%d' % (layerNum + 2)], [-1,self.decoderDataShape[1]/self.encoderNumFilters[-1],self.encoderNumFilters[-1]] )
        for layerNum in range( 1 , len( self.decoderNumFilters ) ):
            absLayerNum = len( self.encoderDataShape ) + layerNum
            outputShape = [tf.shape(self.a['a%d' % (absLayerNum-1)] )[0],self.decoderDataShape[layerNum+1],self.decoderNumFilters[layerNum]]
            self.h['h%d' % (absLayerNum)] = \
                net_ops.layer_deconv1d(self.a['a%d'%(absLayerNum-1)], self.weights['decoder_w%d'%(absLayerNum)], self.biases['decoder_b%d'%(absLayerNum)], outputShape, padding=padding, stride=self.decoderStride[layerNum-1])
            if layerNum < len( self.decoderNumFilters )-1:
                if skipConnect:
                    self.h['h%d' % (absLayerNum)] += self.h['h%d' % (len( self.decoderNumFilters ) - layerNum - 1)]
                self.a['a%d' % (absLayerNum)] = net_ops.layer_activation(self.h['h%d' % (absLayerNum)], activationFunc)
            else:
                if skipConnect:
                    self.h['h%d' % (absLayerNum)] += self.a['a0']
                self.a['a%d' % (absLayerNum)] = net_ops.layer_activation(self.h['h%d' % (absLayerNum)], 'linear')

        # output of final layer
        self.y_recon = tf.squeeze( self.a['a%d' % (absLayerNum)] , axis=2)

    def add_train_op(self,name,lossFunc='SSE',learning_rate=1e-3, decay_steps=None, decay_rate=None, piecewise_bounds=None, piecewise_values=None,
             method='Adam', wd_lambda=0.0 ):
        """ Constructs a loss op and training op from a specific loss function and optimiser. User gives the train op a name, and the train op
            and loss opp are stored in a dictionary under that name
        - input:
            name: (str) Name of the training op (to refer to it later in-case of multiple training ops).
            lossFunc: (str) Reconstruction loss function
            learning rate: (float)
            decay_steps: (int) epoch frequency at which to decay the learning rate.
            decay_rate: (float) fraction at which to decay the learning rate.
            piecewise_bounds: (int list) epoch step intervals for decaying the learning rate. Alternative to decay steps.
            piecewise_values: (float list) rate at which to decay the learning rate at the piecewise_bounds.
            method: (str) optimisation method.
            wd_lambda: (float) scalar to control weighting of weight decay in loss.

        """
        # construct loss op
        self.train_ops['%s_loss'%name] = net_ops.loss_function_reconstruction_1D(self.y_recon, self.y_target, func=lossFunc)

        # weight decay loss contribution
        wdLoss = net_ops.loss_weight_decay(wd_lambda)

        # construct training op
        self.train_ops['%s_train'%name] = \
            net_ops.train_step(self.train_ops['%s_loss'%name]+wdLoss, learning_rate, decay_steps, decay_rate, piecewise_bounds, piecewise_values,method)




    def train(self, dataTrain, dataVal, train_op_name, n_epochs, save_addr, visualiseRateTrain=0, visualiseRateVal=0, save_epochs=[1000]):
        """ Calls network_ops function to train a network.
        - input:
            dataTrain: (obj) iterator object for training data.
            dataVal: (obj) iterator object for validation data.
            train_op_name: (string) name of training op created.
            n_epochs: (int) number of loops through dataset to train for.
            save_addr: (str) address of a directory to save checkpoints for desired epochs.
            visualiseRateTrain: (int) epoch rate at which to print training loss in console
            visualiseRateVal: (int) epoch rate at which to print validation loss in console
            save_epochs: (int list) epochs to save checkpoints at.
        """

        net_ops.train( self, dataTrain, dataVal, train_op_name, n_epochs, save_addr, visualiseRateTrain, visualiseRateVal, save_epochs )


    def add_model(self,addr,name):

        self.modelsAddrs[name] = addr

    def encoder( self, modelName, dataSamples  ):
        """ Extract the latent variable of some dataSamples using a trained model
        - input:
            modelName: (str) Name of the model to use (previously added with add_model() )
            dataSample: (array) Shape [numSamples x numBands]
        - output:
            dataZ: (array) Shape [numSamples x arbitrary]

        """

        with tf.Session() as sess:

            # load the model
            net_ops.load_model(self.modelsAddrs[modelName], sess)

            # get latent values
            dataZ = sess.run(self.z, feed_dict={self.x: dataSamples})

            return dataZ



    def decoder( self, modelName, dataZ  ):
        """ Extract the reconstruction of some dataSamples (with latent representation) using a trained model
        - input:
            modelName: (str) Name of the model to use (previously added with add_model() )
            dataZ: (array) Latent representation of data samples. Shape [numSamples x arbitrary]
        - output:
            dataY_recon: (array) Reconstructed data. Shape [numSamples x arbitrary]

        """

        with tf.Session() as sess:

            # load the model
            net_ops.load_model(self.modelsAddrs[modelName], sess)

            # get reconstruction
            dataY_recon = sess.run(self.y_recon, feed_dict={self.z: dataZ})

            return dataY_recon


    def encoder_decoder( self, modelName, dataSamples  ):
        """ Extract the reconstruction of some dataSamples using a trained model
        - input:
            modelName: (str) Name of the model to use (previously added with add_model() )
            dataSample: (array) Shape [numSamples x numBands]
        - output:
            dataY_recon: (array) Reconstructed data. Shape [numSamples x arbitrary]

        """

        with tf.Session() as sess:

            # load the model
            net_ops.load_model(self.modelsAddrs[modelName], sess)

            # get reconstruction
            dataY_recon = sess.run(self.y_recon, feed_dict={self.x: dataSamples})

            return dataY_recon







