__author__ = 'Oleksandr Lytvynenko'
"""
Class RawDataBuilder simulates raw Data
Contains
========
"""

import numpy as np

DEBUG = True
if DEBUG:
    import matplotlib.pyplot as plt
    from PIL import Image


c = 299792458.0 * 10 ** 9  # micron/s (* 10^12 for nanometers)

class RawDataBuilder(object):
    """A RawDataBuilder.
       """
    Error = ""

    def __init__(self, W, H, Micron_Per_Pixel, lambda0, bandwidth, piezoStep = 0.314,  **kwargs):
        self.W = W
        self.H = H
        self.Micron_Per_Pixel = Micron_Per_Pixel
        self.lambda0 = lambda0
        self.bandwidth = bandwidth
        self.piezoStep = piezoStep
        self.I0 = 100
        self.I0fiber = 95
        self.rfl = 0.99
        dnu = c * bandwidth / (lambda0 * lambda0)
        self.Lc = c / (np.pi * dnu)
        self.cameraNoiseSigma = 0.017


        if DEBUG: print "RawDataBuilder.ctor():"
        if DEBUG:
            print "W, H = ", self.W, self.H
            print "Micron_Per_Pixel = ", self.Micron_Per_Pixel
            print "lambda0, bw = ", self.lambda0, self.bandwidth
            print "dnu, Lc = ", dnu, self.Lc
    pass


    def rawSphere(self, center, R, piezoStart , piezoEnd , whitlight=True, fi0=0, z0=0, piezoRange =None, frameCallback=None):
        """
        :param center: tuple, microns
        :param R: in microns
        :param z0: initial position, microns
        :param fi0: initial phase
        :param piezoRange: an array of piezo positions
        :param frameCallback - a callback functtion called for every generated frame
        :return:
        """
        x0 = center[0] / self.Micron_Per_Pixel ; y0 = center[1] / self.Micron_Per_Pixel ;
        if DEBUG:
            print "z0 = ", z0, "x0 = ", x0, "y0 = ", y0
            print "R = {}, piezoStart = {}, piezoEnd ={} ".format(R, piezoStart, piezoEnd)
        R2 = R * R;
        x = np.arange(0, self.W, 1) - self.W / 2
        y = np.arange(0, self.H, 1) - self.H / 2
        xx, yy = np.meshgrid(x, y)
        if piezoRange != None:
            prng = piezoRange
        else:
            zN =  (np.abs(piezoEnd - piezoStart) / self.piezoStep) + 1
            print "zN = ", zN
            # piezoRange = np.linspace(piezoStart, piezoEnd, num=zN)
            piezoRange = np.linspace(piezoEnd, piezoStart, num=zN)
            print "piezoRange = ", piezoRange

        for z in piezoRange:
            print "z = ", z
            D2 = (xx ** 2 + yy ** 2)  *self.Micron_Per_Pixel*self.Micron_Per_Pixel # microns, distance from center
            dZ =  z + z0 + (R - np.sqrt(R2 - D2))
            q2 = (dZ/self.Lc) ** 2
            npI = self.I0 * np.sqrt(1 - D2 / R2)
            if whitlight:
                npInt = npI * (1 + self.rfl  * np.exp(-4 * q2) * np.cos(4.0 * np.pi * dZ / self.lambda0 - fi0))
            else:
                npInt = npI * (1 + self.rfl * np.cos(4.0 * np.pi * dZ / self.lambda0 - fi0))
            if self.cameraNoiseSigma > 0:
                npInt = npInt * np.random.normal(loc=1.0, scale=self.cameraNoiseSigma,
                                                 size=(self.H, self.W))  # camera noise
            if frameCallback != None:
                a = np.array(npInt,  dtype=np.uint8, copy=True )
                frameCallback(a, z)
        pass

    def singleFiberFlat(self,center,  Rferrule, Dfiber, Hfiber, piezoStart , piezoEnd , whitlight=True, fi0=0, z0=0, piezoRange =None, frameCallback=None):
        """
        :param center: tuple, microns
        :param z0: initial position, microns
        :param fi0: initial phase
        :param piezoRange: an array of piezo positions
        :param frameCallback - a callback functtion called for every generated frame
        :return:
        """

        x0 = center[0] / self.Micron_Per_Pixel; # pix
        y0 = center[1] / self.Micron_Per_Pixel; # pix
        # Rfbr = Rfiber / self.Micron_Per_Pixel
        Rf2 = Rferrule * Rferrule  ;  # microns
        if DEBUG:
            print "z0 = ", z0, "x0 = ", x0, "y0 = ", y0
            print "Rferrule = {}, Dfiber = {}, Hfiber ={} ".format(Rferrule, Dfiber, Hfiber   )
        x = np.arange(0, self.W, 1) - x0   #   pix
        y = np.arange(0, self.H, 1) - y0   # pix
        xx, yy = np.meshgrid(x, y)   # pix
        if piezoRange != None:
            prng =piezoRange
        else:
            zN =int( np.abs (piezoEnd - piezoStart) / self.piezoStep) +1
            piezoRange = np.linspace(piezoEnd, piezoStart, num=zN)
            piezoRange = piezoRange * np.random.normal(loc=1.0, scale=self.piezoStep/24, size=zN) #camera noise
        for z in piezoRange:  #micron
            Dst2 = (xx ** 2 + yy ** 2)  *self.Micron_Per_Pixel*self.Micron_Per_Pixel # microns, distance from center
            r=np.sqrt(Dst2)  # microns
            dZ = (z + z0 + (Rferrule - np.sqrt(Rf2 - Dst2))) - ( r < Dfiber/2  ) * Hfiber # microns
            q2 = (dZ / self.Lc) ** 2
            npI = self.I0 * ( np.sqrt(1 - Dst2 / Rf2)) * (r> Dfiber/2 ) +  self.I0fiber * (r <= Dfiber/2 )
            if whitlight:
                npInt = npI * (1 + self.rfl * np.exp(-4 * q2) * np.cos(4.0 * np.pi * dZ / self.lambda0 - fi0))
            else:
                npInt = npI * (1 + self.rfl * np.cos(4.0 * np.pi * dZ / self.lambda0 - fi0))
            if self.cameraNoiseSigma>0:
                npInt = npInt * np.random.normal(loc=1.0, scale=self.cameraNoiseSigma, size=(self.H,self.W)) #camera noise
            if frameCallback != None:
                a = np.array(npInt, dtype=np.uint8, copy=True)
                frameCallback(a, z)
        pass

