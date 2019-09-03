'''
    Description: various functions for deep learning built on-top of tensorflow. The high-level modules in the package
    call these functions.

    - File name: network_ops.py
    - Author: Lloyd Windrim
    - Date created: June 2019
    - Python package: deephyp

'''

import tensorflow as tf
import math
import numpy as np
from os.path import join, exists, basename, split
import json

def create_variable(shape,method='gaussian',wd=False):
    """ Setup a trainable variable (collection of parameters) of a particular shape.

    Args:
        shape (list): Data shape.
        method (str): How to initialise parameter values.
        wd (boolean): Setup weight decay for this variable.

    Returns:
        (tensor): Set of parameters for the given variable.
    """
    return tf.Variable(init_weight(method, shape, wd=wd))


def layer_fullyConn(input, W, b):
    """ Define a fully connected layer operation. Also called a 'dense' layer.

    Args:
        input (tensor): Data input into the layer. Shape [numSamples x numInputNeurons].
        W (tensor): Weight parameters for the layer. Shape [numInputNeurons x numOutputNeurons].
        b (tensor): Bias parameters for the layer. Shape [numOutputNeurons].

    Returns:
        (tensor): Computes layer output. Shape [numSamples x numOutputNeurons].
    """
    return tf.matmul(input, W) + b

def layer_conv1d(input, W, b, stride=1,padding='SAME'):
    """ Define a 1 dimensional convolution layer operation.

    Args:
        input (tensor): Data input into the layer. Shape [numSamples x numInputNeurons x numFiltersIn].
        W (tensor): Weight parameters of the filters/kernels. Shape [filterSize x numFiltersIn x numFiltersOut].
        b (tensor): Bias parameters for the layer. Shape [numFiltersOut].
        stride (int): Stride at which to convolve (must be >= 1).
        padding (str): Type of padding to use ('SAME' or 'VALID').

    Returns:
        (tensor): Computes layer output. Shape [numSamples x numOutputNeurons x numFiltersOut].
    """

    if (padding!='SAME')&(padding!='VALID'):
        raise ValueError('unknown padding type: %s. Use SAME or VALID' % padding)
    if stride < 1:
        raise ValueError('stride must be greater than 0. Stride = %d found in conv layer.'% stride)

    return tf.nn.conv1d(input,W,stride=stride,padding=padding) + b


def layer_deconv1d(input, W, b, outputShape, stride=1,padding='SAME'):
    """ Define a 1 dimensional deconvolution layer operation. Also called convolutional transpose or upsampling layer.

    Args:
        input (tensor): Data input into the layer. Shape [numSamples x numInputNeurons x numFiltersIn].
        W (tensor): Weight parameters of the filters/kernels. Shape [filterSize x numFiltersOut x numFiltersIn].
        b (tensor): Bias parameters for the layer. Shape [numFiltersOut].
        outputShape (list): Expected shape of the layer output. Shape [numSamples x numOutputNeurons x numFiltersOut].
        stride (int): Stride at which to convolve (must be >= 1).
        padding (str): Type of padding to use ('SAME' or 'VALID').

    Returns:
        (tensor): Computes layer output. Shape [numSamples x numOutputNeurons x numFiltersOut].
    """

    if (padding!='SAME')&(padding!='VALID'):
        raise ValueError('unknown padding type: %s. Use SAME or VALID' % padding)
    if stride < 1:
        raise ValueError('stride must be greater than 0. Stride = %d found in deconv layer.'% stride)

    return tf.nn.conv1d_transpose(input,W,outputShape,strides=stride,padding=padding) + b



def layer_activation(input, func='sigmoid'):
    """ Define an activation function operation.

    Args:
        input (tensor): Data input into the function.
        func (str): Type of activation function. (relu, sigmoid, linear).

    Returns:
        (tensor): Computes activation. Shape is same as input.
    """

    if func == 'relu':
        a = tf.nn.relu(input)
    elif func == 'sigmoid':
        a = tf.nn.sigmoid(input)
    elif func == 'linear':
        a = input
    else:
        raise ValueError('unknown activation function: %s. Use relu, sigmoid or linear.' % func)

    return a

def conv_output_shape(inputShape, filterSize, padding, stride):
    """ Computes the expected output shape (for the convolving axis only) of a convolution layer given an input shape.

    Args:
        inputShape (int): Shape of convolving axis of input data.
        filterSize (int): Size of filter/kernel of convolution layer.
        stride (int): Stride at which to convolve (must be >= 1).
        padding (str): Type of padding to use ('SAME' or 'VALID').

    Returns:
        (int): Output shape of convolving axis for given layer and input shape.
    """
    if padding=='VALID':
        outputShape = np.ceil( (inputShape - (filterSize-1))/stride )
    elif padding=='SAME':
        outputShape = np.ceil(inputShape / stride)
    else:
        raise ValueError('unknown padding type: %s. Use SAME or VALID' % padding)

    return int(outputShape)


def train_step(loss, learning_rate=1e-3, decay_steps=None, decay_rate=None, piecewise_bounds=None, piecewise_values=None,
             method='Adam'):
    """ Operation for training the weights of the network by optimising them to minimise the loss function. Note that \
        the default is a constant learning rate (no decay).

    Args:
        loss (tensor): Output of network loss function.
        learning_rate: (float) Controls the degree to which the weights are updated during training.
        decay_steps (int): Epoch frequency at which to decay the learning rate.
        decay_rate (float): Fraction at which to decay the learning rate.
        piecewise_bounds (int list): Epoch step intervals for decaying the learning rate. Alternative to decay steps.
        piecewise_values (float list): Rate at which to decay the learning rate at the piecewise_bounds.
        method (str): Optimisation method. (Adam, SGD).

    Returns:
        (op) A train op.
    """


    global_step = tf.Variable(0, trainable=False, name='global_step')

    # update learning rate for current step
    if decay_rate != None:
        lr = tf.train.exponential_decay(learning_rate,
                                        global_step,
                                        decay_steps,
                                        decay_rate, staircase=True)
    elif piecewise_bounds != None:
        lr = tf.train.piecewise_constant(global_step, piecewise_bounds, [learning_rate] + piecewise_values)
    else:
        lr = learning_rate


    if method == 'Adam':
        optimizer = tf.train.AdamOptimizer(lr)
    elif method == 'SGD':
        optimizer = tf.train.GradientDescentOptimizer(lr)
    else:
        raise ValueError('unknown optimisation method: %s. Use Adam or SGD.' % method)

    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def loss_function_reconstruction_1D(y_reconstructed,y_target,func='SSE'):
    """ Reconstruction loss function op, comparing 1D tensors for network reconstruction and target.

    Args:
        y_reconstructed (tensor): Output of network (reconstructed 1D vector). Shape [numSamples x inputSize].
        y_target (tensor): What the network is trying to reconstruct (1D vector). Shape [numSamples x inputSize].
        func (string): The name of the loss function to be used. 'SSE'-sum of square errors,'CSA'-cosine spectral angle, \
            'SA'-spectral angle, 'SID'-spectral information divergence.

    Returns:
        (tensor): Reconstruction loss.
    """
    if func == 'SSE':
        # sum of squared errors loss
        loss = tf.reduce_sum( tf.square(y_target - y_reconstructed) )

    elif func == 'CSA':
        # cosine of spectral angle loss
        normalize_r = tf.math.l2_normalize(tf.transpose(y_reconstructed),axis=0)
        normalize_t = tf.math.l2_normalize(tf.transpose(y_target),axis=0)
        loss = tf.reduce_sum( 1 - tf.reduce_sum(tf.multiply(normalize_r, normalize_t),axis=0 ) )

    elif func == 'SA':
        # spectral angle loss
        normalize_r = tf.math.l2_normalize(tf.transpose(y_reconstructed),axis=0)
        normalize_t = tf.math.l2_normalize(tf.transpose(y_target),axis=0)
        loss = tf.reduce_sum( tf.math.acos(tf.reduce_sum(tf.multiply(normalize_r, normalize_t),axis=0 ) ) )

    elif func == 'SID':
        # spectral information divergence loss
        t = tf.divide( tf.transpose(y_target) , tf.reduce_sum(tf.transpose(y_target),axis=0) )
        r = tf.divide( tf.transpose(y_reconstructed) , tf.reduce_sum(tf.transpose(y_reconstructed),axis=0) )
        loss = tf.reduce_sum( tf.reduce_sum( tf.multiply(t,tf.log(tf.divide(t,r))) , axis=0)
                              + tf.reduce_sum( tf.multiply(r,tf.log(tf.divide(r,t))) , axis=0) )
    else:
        raise ValueError('unknown loss function: %s. Use SSE, CSA, SA or SID.' % func)

    return loss


def loss_function_crossentropy_1D( y_pred, y_target, class_weights=None, num_classes=None):
    """ Cross entropy loss function op, comparing 1D tensors for network prediction and target. Weights the classes \
        when calculating the loss to balance un-even training batches. If class weights are not provided, then no \
        weighting is done (weight of 1 assigned to each class).

    Args:
        y_pred (tensor): Output of network (1D vector of class scores). Shape [numSamples x numClasses].
        y_target (tensor): One-hot classification labels (1D vector). Shape [numSamples x numClasses].
        class_weights (tensor): Weight for each class. Shape [numClasses].
        num_classes (int):

    Returns:
        (tensor): Cross-entropy loss.
    """

    if class_weights==None:
        class_weights = tf.constant(1,shape=[num_classes],dtype=tf.dtypes.float32)

    sample_weights = tf.reduce_sum( tf.multiply(y_target, class_weights ), axis=1) # weight of each sample
    loss = tf.reduce_mean( tf.losses.softmax_cross_entropy(
        onehot_labels=y_target,logits=y_pred,weights=sample_weights ) )

    return loss


def loss_weight_decay(wdLambda):
    """ Weight decay loss op, regularises network by penalising parameters for being too large.

    Args:
        wdLambda (float): Scalar to control weighting of weight decay in loss.

    Returns:
        (tensor) : Weight-decay loss.
    """

    return tf.multiply( wdLambda , tf.reduce_sum(tf.get_collection('wd')) )

def balance_classes(y_target,num_classes):
    """ Calculates the class weights needed to balance the classes, based on the number of samples of each class in the \
        batch of data.

    Args:
        y_target (tensor): One-hot classification labels (1D vector). Shape [numSamples x numClasses]
        num_classes (int):

    Returns:
        (tensor): A weighting for each class that balances their contribution to the loss. Shape [numClasses].
    """
    y_target = tf.reshape( y_target, [-1, num_classes] )
    class_count = tf.add( tf.reduce_sum( y_target, axis=0 ), tf.constant( [1]*num_classes, dtype=tf.float32 ) )
    class_weights = tf.multiply( tf.divide( tf.ones( ( 1, num_classes) ), class_count ), tf.reduce_max( class_count ) )

    return class_weights


def save_model(addr,sess,saver,current_epoch,epochs_to_save):
    """Saves a checkpoint at a list of epochs.

    Args:
        addr (str): Address of a directory to save checkpoint for current epoch.
        sess (obj): Tensor flow session object.
        saver (obj): Tensor flow save object.
        current_epoch (int): The current epoch.
        epochs_to_save (int list): Epochs to save checkpoints at.

    """

    if current_epoch in epochs_to_save:
        saver.save(sess, join(addr,"epoch_%i"%(current_epoch),"model.ckpt"))


def load_model(addr,sess):
    """Loads a model from the address of a checkpoint.

    Args:
        addr (str): Address of a directory to save checkpoint for current epoch.
        sess (obj): Tensor flow session object.

    """
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, join(addr, 'model.ckpt'))


def save_config(net_obj,addr):
    """Saves a network config file. Saves the variables listed in net_config within the network object.

    Args:
        net_obj (obj): Network object.
        addr (obj): Directory of where to store the config.json file.

    """

    data = {}
    for config_parameter in net_obj.net_config:
        data[config_parameter] = getattr(net_obj,config_parameter)

    with open(join(addr,'config.json'), 'w') as outfile:
        json.dump(data, outfile)

def load_config(net_obj,addr):
    """Loads a network config file. Loads from variables in the config.json file and overwrites variables in network \
        object. Applies to variables in the net_config list in the network object.

    Args:
        net_obj (obj): Network object.
        addr (obj): Directory location of config.json file.

    """

    with open(addr, 'r') as outfile:
        data = json.load(outfile)

    for config_parameter in data:
        setattr(net_obj,config_parameter,data[config_parameter])



def train( net_obj , dataTrain, dataVal, train_op_name, n_epochs, save_addr, visualiseRateTrain=0, visualiseRateVal=0,
           save_epochs=[1000] ):
    """ Function for training a network. Updates the network weights through the training op. The function will check \
        the save address for a model checkpoint to load, otherwise it will begin training from scratch.

    Args:
        net_obj (obj): Network object.
        dataTrain (obj): Iterator object for training data.
        dataVal (obj): Iterator object for validation data.
        train_op_name (string): Name of training op created.
        n_epochs (int): Number of loops through dataset to train for.
        save_addr (str): Address of a directory to save checkpoints for desired epochs, or address of saved checkpoint. \
                        If address is for an epoch and contains a previously saved checkpoint, then the network will \
                        start training from there. Otherwise it will be trained from scratch.
        visualiseRateTrain (int): Epoch rate at which to print training loss in console.
        visualiseRateVal (int): Epoch rate at which to print validation loss in console.
        save_epochs (int list): Epochs to save checkpoints at.
    """

    if np.shape(dataTrain.dataSamples)[1] != net_obj.inputSize:
        raise Exception('the data dimensionality must match the network input size. '
                        'Data size: %d, network input size: %d'%(np.shape(dataTrain.dataSamples)[1], net_obj.inputSize))

    batchSize = dataTrain.batchSize
    numSamples = dataTrain.numSamples

    numIters = numSamples // batchSize
    if (numSamples % batchSize)>0:
        numIters+=1

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # check if addr has 'epoch' in name and contains a checkpoint
        if exists(join(save_addr,'checkpoint')) & ('epoch' in basename(save_addr)):
            # load a checkpoint
            saver.restore(sess,join(save_addr,'model.ckpt'))
            epoch_start = int((basename(save_addr)).split('_')[-1]) + 1
            save_addr = split(save_addr)[0]
        else:
            # save directory is empty
            epoch_start = 0
            # create network config file in directory
            save_config(net_obj,save_addr)

        for epoch_i in range(epoch_start, n_epochs+1):
            train_error = []
            for batch_i in range(numIters):
                train_batch_x , train_batch_y = dataTrain.next_batch()

                # update weights and biases
                sess.run(net_obj.train_ops['%s_train'%(train_op_name)], feed_dict={net_obj.x: train_batch_x,
                                                                                   net_obj.y_target: train_batch_y})

                # training loss
                if visualiseRateTrain > 0:
                    if epoch_i % visualiseRateTrain == 0:
                        train_error.append( net_obj.train_ops['%s_loss' % (train_op_name)].eval(
                            {net_obj.x: train_batch_x, net_obj.y_target: train_batch_y}) )

                if batch_i == numIters - 1:
                    dataTrain.reset_batch()

            # outputs average batch error
            if visualiseRateTrain > 0:
                if epoch_i % visualiseRateTrain == 0:
                    train_error = np.array(train_error)
                    print("epoch: %d, training loss: %g" % (epoch_i, np.mean(train_error)))




            # iterate over validation samples and output loss
            if visualiseRateVal > 0:
                if epoch_i % visualiseRateVal == 0:

                    val_error = []
                    for batch_i in range(dataVal.numSamples // dataVal.batchSize):
                        val_batch_x, val_batch_y = dataVal.next_batch()

                        val_error.append( net_obj.train_ops['%s_loss' % (train_op_name)].eval(
                                {net_obj.x: val_batch_x, net_obj.y_target: val_batch_y}) )

                        if batch_i == (dataVal.numSamples // dataVal.batchSize)-1:
                            dataVal.reset_batch()

                    val_error = np.array(val_error)
                    print("epoch: %d, validation loss: %g" % (epoch_i, np.mean(val_error)))

            save_model(save_addr,sess,saver,epoch_i,save_epochs)









def init_weight(opts, shape, stddev=0.1, const=0.1, wd = False, dtype=tf.float32):
    """ Weight initialisation function.

    Args:
        opts (str): Method for initialising variable. ('gaussian','truncated_normal','xavier','xavier_improved', \
            'constant').
        shape (list): Data shape.
        stddev (int): Standard deviation used by 'gaussian' and 'truncated_normal' variable initialisation methods.
        const (int): Constant value to initialise variable to if using 'constant' method.
        wd (boolean): Whether this variable contributes to weight decay or not.
        dtype (tf.dtype): Data type for variable.

    Returns:
        weights:
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
    elif opts == 'constant':
        weights = tf.constant(const, shape)
    else:
        raise ValueError('Unknown weight initialization method %s' % opts)

    # set up weight decay on weights
    if wd:
        weight_decay = tf.nn.l2_loss(weights)
        tf.add_to_collection('wd', weight_decay)

    return weights
