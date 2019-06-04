import tensorflow as tf
import math


class mlp_1D_network():

    def __init__( self , inputSize , encoderSize=[50,30,10] , activationFunc='sigmoid' ,
                  tiedWeights=None , weightInitOpt='truncated_normal' , weightStd=0.1  ):

        """ Class for setting up a 1-D multi-layer perceptron autoencoder network
        - input:
            inputSize: (int) Number of dimensions of input data (i.e. number of spectral bands)
            encoderSize: (int list) Number of nodes at each layer of the encoder. List length is number of encoder layers.
            activationFunc: (string) [sigmoid, ReLU, tanh]
            tiedWeights: (binary list) 1 - tied weights of specific encoder layer to corresponding decoder weights.
                                        0 - do not not weights of specific layer
                                        None - sets all layers to 0
            weightInitOpt: (string) Method of weight initialisation. [gaussian, truncated_normal, xavier, xavier_improved]
            weightStd: (float) Used by 'gaussian' and 'truncated_normal' weight initialisation methods
        """

        self.numBands = inputSize
        self.encoderSize = [inputSize] + encoderSize
        self.decoderSize = self.encoderSize[::-1][1:]
        self.activationFunc = activationFunc
        self.tiedWeights = tiedWeights
        self.weightInitOpt = weightInitOpt
        self.weightStd = weightStd

        x = tf.placeholder("float", [None, inputSize])

        self.weights = { }
        self.biases = { }

        # encoder weights
        for layerNum in range( len( self.encoderSize ) ):
            self.weights['encoder_w%i'%(layerNum+1)] = \
                tf.Variable(init_weight(weightInitOpt,[self.encoderSize[layerNum], self.encoderSize[layerNum+1]]))

        # decoder weights
        for layerNum in range( len( self.decoderSize ) ):
            self.weights['decoder_w%i' % (len( self.encoderSize ) + layerNum + 1)] = \
                tf.Variable(init_weight(weightInitOpt,[self.decoderSize[layerNum], self.decoderSize[layerNum + 1]]))


        # encoder biases
        for layerNum in range( len( self.encoderSize ) ):
            self.biases['encoder_b%i'%(layerNum+1)] = \
                tf.Variable(init_weight(weightInitOpt,[self.encoderSize[layerNum+1]]))

        # decoder biases
        for layerNum in range( len( self.decoderSize ) ):
            self.biases['decoder_b%i' % (len( self.encoderSize ) + layerNum + 1)] = \
                tf.Variable(init_weight(weightInitOpt,[self.decoderSize[layerNum + 1]]))

        # build network using encoder, decoder and x placeholder as input
        self.encoder()
        self.decoder()


    def encoder( self  ):
        self.x
        pass

    def decoder( self  ):
        pass

    # training and inference data will call encoder and decoder and feed into the placeholder x
    def fit( self , trainingData ):
        pass

    def predict( self , inferenceData):
        pass



def init_weight(opts, shape, stddev=0.1, wd = None, dtype=tf.float32):

    """ Weight initialisation function.
    See K.He, X.Zhang, S.Ren, and J.Sun.Delving deep into rectifiers: Surpassing human - level performance
    on imagenet classification.CoRR, (arXiv:1502.01852 v1), 2015.
    """
    if opts == 'gaussian':
        weights = tf.random_normal(shape, stddev=stddev, dtype=dtype)
    elif opts == 'truncated_normal':
        weights = tf.truncated_normal(shape, stddev=stddev)
    elif opts == 'xavier':
        h = shape[0]
        w = shape[1]
        try:
            num_in = shape[2]
        except:
            num_in = 1
        sc = math.sqrt(3.0 / (h * w * num_in))
        weights = tf.multiply(tf.random_normal(shape, dtype=dtype) * 2 - 1, sc)
    elif opts == 'xavier_improved':
        h = shape[0]
        w = shape[1]
        try:
            num_out = shape[3]
        except:
            num_out = 1
        sc = math.sqrt(2.0 / (h * w * num_out))
        weights = tf.multiply(tf.random_normal(shape, dtype=dtype), sc)
    else:
        raise ValueError('Unknown weight initialization method %s' % opts)

    # set up weight decay on weights
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return weights