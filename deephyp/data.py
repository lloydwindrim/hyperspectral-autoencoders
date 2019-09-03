'''
    Description: high-level classes for using hyperspectral data with the deep learning modules.

    - File name: data.py
    - Author: Lloyd Windrim
    - Date created: June 2019
    - Python package: deephyp


'''


import numpy as np


class HypImg:
    """Class for handling data. Stores meta-data and contains attributes for pre-processing the data. If passed labels, \
        samples with label zero are considered as a background class. This class is not included in numClasses and data \
        samples with this label have a one-hot vector label of all zeros.

    Args:
        spectralInput (np.array float): Spectral dataset. Shape can be [numRows x numCols x numBands] or \
            [numSamples x numBands].
        wavelengths (np.array float): Vector of wavelengths that spectralInput wavelengths lie within.
        bands (np.array int): Wavelength indexes for each band of spectralInput. Shape [numBands].
        labels (np.array int): Class labels for each spectral sample in spectralInput. Shape can be [numRows x numCols] \
            or [numSamples].

    Attributes:
        spectra (np.array float): Un-pre-processed spectral data with shape [numSamples x numBands].
        spectraCube (np.array float): If data passed as image - un-pre-processed spectral datacube with \
            shape [numSamples x numBands]. Else None.
        spectraPrep (np.array float): Pre-processed spectral data with shape [numSamples x numBands].
        numSamples (int): The number of spectra.
        numRows (int): If data passed as image - the number of image rows. Else None.
        numCols (int): If data passed as image - the number of image columns. Else None.
        wavelengths (np.array float): If provided - vector of wavelengths that spectra wavelengths lie within. Else None.
        bands (np.array int): If provided - wavelength indexes for each band of spectra with shape [numBands]. Else None.
        labels (np.array int): If provided - class labels for each spectral sample with shape [numSamples]. Else None.
        labelsOnehot (np.array int): If labels provided - the one-hot label vector for each sample. Samples with label \
            zero (background class) have a one-hot vector of all zeros. Else None.
    """


    def __init__( self , spectralInput , labels=None, wavelengths=None, bands=None  ):

        # if input is of shape [numRows x numCols x numBands], convert to [numSamples x numBands]
        if len( spectralInput.shape ) == 3:
            self.numRows , self.numCols , self.numBands = spectralInput.shape
            self.numSamples = self.numRows * self.numCols
            self.spectra = (np.reshape( spectralInput , ( -1, self.numBands ) )).astype(np.float)
            self.spectraCube = spectralInput.astype(np.float)
        else:
            self.numSamples , self.numBands = spectralInput.shape
            self.numRows = None
            self.numCols = None
            self.spectra = spectralInput.astype(np.float)
            self.spectraCube = None

        # if labels provided, determine number of classes and one-hot labels
        if labels is not None:
            if len(labels.shape) == 2:
                self.labels = np.reshape(labels, -1)
            else:
                self.labels = labels
            self.numClasses = len( np.unique(self.labels)[np.unique(self.labels)>0] )

            # create one-hot labels for classes > 0
            self.labelsOnehot = np.zeros((self.numSamples, self.numClasses))
            self.labelsOnehot[np.arange(self.numSamples)[self.labels>0], (self.labels-1)[self.labels>0]] = 1

            self.labels = self.labels[:,np.newaxis]
        else:
            self.labels = None
            self.labelsOnehot = None
            self.numClasses = None


        self.wavelengths = wavelengths
        self.bands = bands

    def pre_process( self , method='minmax' ):
        """Pre-process data for input into the network. Stores in the spectraPrep attribute.

        Args:
            method (str): Method of pre-processing. Current options: 'minmax'
        """
        if method == 'minmax':
            # scales each spectra to be between [0 1] (lower bound is actually a small non-zero number)
            self.spectraPrep = self.spectra - np.transpose(np.tile(np.min(self.spectra,axis=1)-(1e-3),(self.numBands,1)))
            self.spectraPrep = self.spectraPrep / np.transpose(np.tile(np.max(self.spectra, axis=1), (self.numBands, 1)))



class Iterator:
    """ Class for iterating through data, to train the network.

        Args:
            dataSamples (np.array float): Data to be input into the network. Shape [numSamples x numBands].
            targets (np.array int): Network output target of each dataSample. For classification, these are the class \
                labels, and it could be the dataSamples for autoencoders. Shape [numSamples x arbitrary]
            batchSize (int): Number of dataSamples per batch

        Attributes:
            dataSamples (np.array float): Data to be input into the network. Shape [numSamples x numBands].
            targets (np.array int): Network output target of each dataSample. For classification, these are the class \
                labels, and it could be the dataSamples for autoencoders. Shape [numSamples x arbitrary]
            batchSize (int): Number of dataSamples per batch. If None - set to numSamples (i.e. whole dataset).
            numSamples (int): The number of data samples.
            currentBatch (int list): A list of indexes specifying the data samples in the current batch. \
                Shape [batchSize]

    """

    def __init__(self, dataSamples,targets,batchSize=None):

        self.dataSamples = dataSamples
        self.targets = targets
        self.numSamples = np.shape(dataSamples)[0]
        if batchSize is not None:
            self.batchSize = batchSize
        else:
            self.batchSize = self.numSamples
        self.currentBatch = np.arange(self.batchSize)


    def next_batch(self):
        """ Return next batch of samples and targets (with batchSize number of samples). The currentBatch indexes are \
            incremented. If end of dataset reached, the indexes wraps around to the beginning.

        Returns:
            (tuple): 2-element tuple containing:

            - (*np.array float*) - Batch of data samples at currentBatch indexes. Shape [batchSize x numBands].
            - (*np.array int*) - Batch of targets at currentBatch indexes. Shape [batchSize x arbitrary].
        """

        batchData = self.dataSamples[self.currentBatch, :]
        batchTargets = self.targets[self.currentBatch, :]

        # update current batch
        self.currentBatch += self.batchSize
        self.currentBatch[self.currentBatch >= self.numSamples] = \
            self.currentBatch[self.currentBatch >= self.numSamples] - self.numSamples

        return batchData , batchTargets

    def get_batch(self, idx):
        """ Returns a specified set of samples and targets.

        Args:
            idx (int list): Indexes of samples (and targets) to return.
        Returns:
            (tuple): 2-element tuple containing:

            - (*np.array float*) - Batch of data samples at [idx] indexes. Shape [length(idx) x numBands].
            - (*np.array int*) - Batch of targets at [idx] indexes. Shape [length(idx) x arbitrary].
        """

        batchData = self.dataSamples[idx, :]
        batchTargets = self.targets[idx, :]

        return batchData, batchTargets


    def reset_batch(self):
        """ Resets the current batch to the beginning.

        """

        self.currentBatch = np.arange(self.batchSize)

    def shuffle(self):
        """ Randomly permutes all dataSamples (and corresponding targets).

        """
        idx = np.random.permutation(np.shape(self.dataSamples)[0])
        self.dataSamples = self.dataSamples[idx,:]
        self.targets = self.targets[idx,:]









