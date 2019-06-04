import numpy as np


class Data:


    def __init__( self , spectralInput , wavelengths=None, bands=None ):
        """ Class for handling data
        - input:
            dataInput: (numpy array) Spectral value at each band.
                        Shape can be [numRows x numCols x numBands] or [numSamples x numBands].
            wavelengths: (numpy array) Wavelengths spanning those of spectralInput
            bands: (numpy array) Wavelength indexes for each band of spectralInput. Must have size numBands.
        """

        # if input is of shape [numRows x numCols x numBands], convert to [numSamples x numBands]
        if len( spectralInput.shape ) == 3:
            self.numRows , self.numCols , self.numBands = spectralInput.shape
            self.numSamples = self.numRows * self.numCols
            self.spectra = np.transpose( np.reshape( spectralInput , ( self.numBands , -1 ) ) )
            self.spectraCube = spectralInput
        else:
            self.numSamples , self.numBands = spectralInput.shape
            self.numRows = None
            self.numCols = None
            self.spectra = spectralInput
            self.spectraCube = None

        self.wavelengths = wavelengths
        self.bands = bands

    def pre_process( self , method ):
        """
        - input:
            method: (string)
                - radiometric_normalisation
                -

        """
        pass


