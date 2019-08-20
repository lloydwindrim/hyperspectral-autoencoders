'''
    File name: data.py
    Author: Lloyd Windrim
    Date created: June 2019
    Python package: deephyp

    Description: high-level classes for using hyperspectral data with the deep learning modules.

'''


import numpy as np


class HypImg:


    def __init__( self , spectralInput , wavelengths=None, bands=None ):
        """ Class for handling data.
        - input:
            dataInput: (array) Spectral value at each band.
                        Shape can be [numRows x numCols x numBands] or [numSamples x numBands].
            wavelengths: (numpy array) Wavelengths spanning those of spectralInput
            bands: (array) Wavelength indexes for each band of spectralInput. Must have size numBands.
        """

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

        self.wavelengths = wavelengths
        self.bands = bands

    def pre_process( self , method ):
        """Pre-process data for input into the network. Stores in spectraPrep.
        - input:
            method: (string)
                - minmax
                -

        """
        if method == 'minmax':
            # scales each spectra to be between [0 1] (lower bound is actually a small non-zero number)
            self.spectraPrep = self.spectra - np.transpose(np.tile(np.min(self.spectra,axis=1)-(1e-3),(self.numBands,1)))
            self.spectraPrep = self.spectraPrep / np.transpose(np.tile(np.max(self.spectra, axis=1), (self.numBands, 1)))


class Iterator:

    def __init__(self, dataSamples,targets,batchSize=None):
        """ Class for iterating through data, to train network.
        - input:
            dataSamples: (array) Spectral value at each band.
                        Shape [numSamples x numBands].
            batchSize: (int) Number of dataSamples per batch
            targets: (array) Target of each dataSample.
                        Shape [numSamples x arbitrary]
        """

        self.dataSamples = dataSamples
        self.targets = targets
        self.numSamples = np.shape(dataSamples)[0]
        if batchSize is not None:
            self.batchSize = batchSize
        else:
            self.batchSize = self.numSamples
        self.currentBatch = np.arange(self.batchSize)

    def next_batch(self):
        """ Return next batch of samples and targets of (with batchSize number of samples).
            If end of dataset reached, it wraps around to the begining.
        - output:
            batchData: (array) [batchSize x numBands]
            batchTargets: (array) [batchSize x arbitrary]
        """

        batchData = self.dataSamples[self.currentBatch, :]
        batchTargets = self.targets[self.currentBatch, :]

        # update current batch
        self.currentBatch += self.batchSize
        self.currentBatch[self.currentBatch >= self.numSamples] = \
            self.currentBatch[self.currentBatch >= self.numSamples] - self.numSamples

        return batchData , batchTargets

    def get_batch(self, idx):
        """ Returns a specific set of samples and targets .
        - input:
            idx: (array) Indexes of samples (and targets) to return.
                        Shape [numSamples].
        - output:
            batchData: (array) [len(idx) x numBands]
            batchTargets: (array) [len(idx) x arbitrary]
        """

        batchData = self.dataSamples[idx, :]
        batchTargets = self.targets[idx, :]

        return batchData, batchTargets


    def reset_batch(self):
        """ Resets the current batch to the begining

        """

        self.currentBatch = np.arange(self.batchSize)

    def shuffle(self):
        """ Randomly permutes the data samples (and corresponding targets)

        """
        idx = np.random.permutation(np.shape(self.dataSamples)[0])
        self.dataSamples = self.dataSamples[idx,:]
        self.targets = self.targets[idx,:]









