import tensorflow as tf
import math
import numpy as np
from os.path import join, exists, basename, split

def create_variable(shape,method='gaussian',wd=False):
    return tf.Variable(init_weight(method, shape, wd=wd))


def layer_fullyConn(input, W, b):
    return tf.matmul(input, W) + b

def layer_conv1d():
    pass



def layer_activation(input, func='sigmoid'):

    if func == 'relu':
        a = tf.nn.relu(input)
    elif func == 'sigmoid':
        a = tf.nn.relu(input)
    elif func == 'linear':
        a = input
    else:
        pass

    return a


def train_step(loss, learning_rate=1e-3, decay_steps=None, decay_rate=None, piecewise_bounds=None, piecewise_values=None,
             method='Adam'):
    """ Op for learning the weights of the network by optimising them to minimise the loss function. Note that the
        default is a constant rate (no decay).
    - input:
        loss: (op) output of network loss function.
        learning_rate: (float)
        decay_steps: (int) epoch frequency at which to decay the learning rate.
        decay_rate: (float) fraction at which to decay the learning rate.
        piecewise_bounds: (int list) epoch step intervals for decaying the learning rate. Alternative to decay steps.
        piecewise_values: (float list) rate at which to decay the learning rate at the piecewise_bounds.
        method: (str) optimisation method.
    - output:
        train_op (op)
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
        'unknown method for optimisation'

    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def loss_function_reconstruction_1D(y_reconstructed,y_target,func='SSE'):
    """ Reconstruction loss function op, comparing 1D tensors for network recontruction and target
    - input:
        y_reconstructed: (tensor) output of network (reconstructed 1D vector). Shape [numSamples x numBands]
        y_target: (tensor) what the network is trying to reconstruct (1D vector). Shape [numSamples x numBands]
        func: (string) the name of the loss function to be used. 'SSE'-sum of square errors,'CSA'-cosine spectral angle,
            'SA'-spectral angle, 'SID'-spectral information divergence.
    - output:
        loss (op)
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

    return loss


def loss_weight_decay(wdLambda):
    """ Weight decay loss op, regularises network by penalising parameters for being too large.
    - input:
        wdLambda: (float) scalar to control weighting of weight decay in loss.
    - output:
        loss (op)
    """

    return tf.multiply( wdLambda , tf.reduce_sum(tf.get_collection('wd')) )


def save_model(addr,sess,saver,current_epoch,epochs_to_save):
    """
    Saves a checkpoint at a list of epochs.
    -input:
        addr: (str) address of a directory to save checkpoint for current epoch.
        sess: (obj) tensor flow session object.
        saver: (obj) tensor flow save object.
        current_epoch: (int)
        epochs_to_save: (int list) epochs to save checkpoints at.

    """

    if current_epoch in epochs_to_save:
        saver.save(sess, join(addr,"epoch_%i"%(current_epoch),"model.ckpt"))


def load_model(addr,sess):
    """
    Loads a model from the address of a checkpoint.
    -input:
        addr: (str) address of a directory to save checkpoint for current epoch.
        sess: (obj) tensor flow session object.

    """
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, join(addr, 'model.ckpt'))


def train( net_obj , dataTrain, dataVal, train_op_name, n_epochs, save_addr, visualiseRateTrain=0, visualiseRateVal=0, save_epochs=[1000] ):
    """ Function for training a network. Updates the network weights through the training op. The function will check the save address
        for a model checkpoint to load, otherwise it will begin training from scratch.
    - input:
        dataTrain: (obj) iterator object for training data.
        dataVal: (obj) iterator object for validation data.
        train_op_name: (string) name of training op created.
        n_epochs: (int) number of loops through dataset to train for.
        save_addr: (str) address of a directory to save checkpoints for desired epochs.
        visualiseRateTrain: (int) epoch rate at which to print training loss in console.
        visualiseRateVal: (int) epoch rate at which to print validation loss in console.
        save_epochs: (int list) epochs to save checkpoints at.
    """

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

        for epoch_i in range(epoch_start, n_epochs+1):
            train_error = []
            for batch_i in range(numIters):
                train_batch_x , train_batch_y = dataTrain.next_batch()

                # update weights and biases
                sess.run(net_obj.train_ops['%s_train'%(train_op_name)], feed_dict={net_obj.x: train_batch_x, net_obj.y_target: train_batch_y})

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
    See K.He, X.Zhang, S.Ren, and J.Sun.Delving deep into rectifiers: Surpassing human - level performance
    on imagenet classification.CoRR, (arXiv:1502.01852 v1), 2015.
    wd - whether this variable contributes to weight decay or not
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
