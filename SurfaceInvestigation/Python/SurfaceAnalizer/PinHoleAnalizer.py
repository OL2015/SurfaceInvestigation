""" PinHoleAnalizer

Contains
========
PinHoleAnalizer

"""
import os
import numpy as np
from scipy import optimize as op
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy import signal
from collections import OrderedDict
import FiberHelpers as fh
from collections import OrderedDict
PIXELSIZE = 1 # in bytes
DEBUG = True



class PinHoleAnalizer(object):
    """A PinHoleAnalizer.
       """
    LogsDirectory =""
    Error = ""
    DimFullPath = ""
    SurfaceFullPath = ""
    FrameWidth = 0
    FrameHeight  = 0
    FrameSize = 0
    def __init__(self, dimPath, surfPath):
        self.DimFullPath = dimPath
        self.SurfaceFullPath = surfPath
        self.LogsDirectory = os.path.dirname(os.path.abspath(dimPath))
        if DEBUG: print "PinHoleAnalizer.ctor():"
        if DEBUG: print " DimFullPath = ", self.DimFullPath
        if DEBUG: print " SurfaceFullPath = ", self.SurfaceFullPath
        if DEBUG: print " LogsDirectory = ", self.LogsDirectory
        self.readFrameSize()
        pass

    def readFrameSize(self):
        print "self.DimFullPath", self.DimFullPath
        import ConfigParser
        config = ConfigParser.RawConfigParser()
        config.read(self.DimFullPath)

        self.FrameWidth = int(config.get('System', 'Width'))
        self.FrameHeight = int(config.get('System', 'Height'))
        print self.FrameWidth, self.FrameHeight
        self.FrameSize = self.FrameWidth * self.FrameHeight
        filesize = os.path.getsize(self.SurfaceFullPath)
        f = open(self.SurfaceFullPath, "rb")
        f.seek(0, 2)  # move the cursor to the end of the file
        sz = f.tell()
        assert(filesize == sz)
        self.frmCounter = filesize / (PIXELSIZE * self.FrameSize )

    def readFrameSizeOld(self):
        with open(self.DimFullPath) as f:
            for line in f:
                spl = line.strip().split('x')
                self.FrameWidth = int(spl[0]);
                self.FrameHeight = int(spl[1])
        self.FrameSize = self.FrameWidth * self.FrameHeight
        filesize = os.path.getsize(self.SurfaceFullPath)
        f = open(self.SurfaceFullPath, "rb")
        f.seek(0, 2)  # move the cursor to the end of the file
        sz = f.tell()
        assert (filesize == sz)
        self.frmCounter = filesize / (PIXELSIZE * self.FrameSize)

    def readPiezoDistanse(self):
        self.PiezoDistanse =  np.zeros((self.frmCounter ), dtype=float)
        peizoDistancePath = self.LogsDirectory +"\\..\\Logs\\piezoDistance.txt"
        self.PiezoDistanse = np.loadtxt(peizoDistancePath, delimiter="\t")
        print "self.PiezoDistanse = ", self.PiezoDistanse

    def getFrame (self, frameNum =0 ):
        f = open(self.SurfaceFullPath, "rb")
        try:
           f.seek(frameNum * (PIXELSIZE * self.FrameSize), 0)  # set cursor to the beginning of the frame
           frameArr = np.fromfile(f, dtype=np.uint8, count=PIXELSIZE * self.FrameSize)
           img = Image.frombytes('L', [self.FrameWidth, self.FrameHeight], frameArr)
        finally:
            f.close()

        plt.close("all")
        plt.gray()
        plt.hist (frameArr, bins=256 )
        plt.show()
        from cStringIO import StringIO
        buffer_ = StringIO()
        plt.savefig(buffer_, format="png")
        buffer_.seek(0)
        imageHist = Image.open(buffer_)
        from PIL.ImageQt import ImageQt
        qimage = ImageQt(imageHist)
        buffer_.close()
        return img, qimage

    def removePatternNoise (self, calibrationMatrixPath, outDataPath):
        # read Calibration Matrix
        f = open(calibrationMatrixPath, "rb")
        try:
            # f.seek(  4)  # set cursor to the beginning of the frame
            cameraSN =  np.fromfile(f, dtype = np.uint64 , count =1 ).byteswap()
            height = np.fromfile(f, dtype = np.uint32 , count =1 ).byteswap()
            width = np.fromfile(f, dtype = np.uint32 , count =1).byteswap()
            print cameraSN, height, width
            A = np.fromfile(f, dtype = np.float , count =height *width ).byteswap()
            print  A[0:255]
            B = np.fromfile(f, dtype=np.float, count=height * width).byteswap()
            print  B[0:255]
        finally:
            f.close()
        # Align surf raw  data
        g = open(self.SurfaceFullPath, "rb")
        gout = open(outDataPath, "wb")
        try:
            for frame in range(self.frmCounter - 1):
                # f.seek(frame * (PIXELSIZE * self.FrameSize), 0)  # set cursor to the beginning of the frame
                Z = np.fromfile(g, dtype=np.uint8, count=self.FrameSize)
                ZR = Z * (1-A) -B
                ZR [ZR>255] = 255 ; ZR[ZR <0] = 0
                Zout = ZR.astype(np.int8)
                # print  Z[0:255]
                # print  Zout[0:255]
                # Zout.tofile(gout)
                gout.write(Zout)
        finally:
            gout.close()
            g.close()
        print  Z[0:255]
        print  Zout[0:255]
        return

    def sliceRawData(self, x, y, halfWidth = 0,  halfHeight = 0):
        width = halfWidth * 2 + 1; height = halfHeight* 2 + 1
        self.WidePinHole = np.zeros(( self.frmCounter ,width, height), dtype = np.uint8)
        print self.WidePinHole.shape
        frame = 0
        f = open(self.SurfaceFullPath, "rb")
        try:
            for frame in range (self.frmCounter -1):
                f.seek(frame * (PIXELSIZE * self.FrameSize ), 0) # set cursor to the beginning of the frame
                shift = ((y - halfHeight) * self.FrameWidth + (x - halfWidth))* PIXELSIZE # shift from the beginning of the frame
                for z in range (0, height ):
                    f.seek(shift, 1) # seek from current position
                    for k in range(0, width):
                        b = np.fromfile(f, dtype = np.uint8, count =1 )
                        self.WidePinHole [frame, z, k] = b
                    shift = shift + (self.FrameWidth -width)* PIXELSIZE # beginning of the next row
        finally:
            f.close()
        return self.WidePinHole

    def drawCurrentPinhole(self, frameNo):
        """ returns ImageQt image for current pinhole"""
        from scipy import signal
        # print self.WidePinHole[:, 0, 0 ]
        sig =  self.WidePinHole[0:300, 0, 0 ] - np.mean(self.WidePinHole[:, 0, 0 ])
        evp = signal.hilbert(sig )
        envelope = np.abs(evp)
        # envelope = np.real( evp)
        print "evp max =", np.max(envelope), "arg max =", np.argmax(envelope)
        plt.close("all")
        print "drawCurrentPinhole"
        # plt.gray()
        # ax = plt.gca()
        z = plt.axvline(x=frameNo,linewidth=2, color='r',  marker='o', markerfacecolor='r')
        # ax.add_artist(z)
        plt.plot(envelope,  color='r' )
        plt.plot(-envelope, color='g')
        plt.plot(sig, color='b',  marker='o', markerfacecolor='r')
        # plt.plot(self.WidePinHole[:, 1, 0])
        plt.show()
        from cStringIO import StringIO
        buffer_ = StringIO()
        plt.savefig(buffer_, format="png")
        buffer_.seek(0)
        image = Image.open(buffer_)
        from PIL.ImageQt import ImageQt
        qimage = ImageQt(image)
        buffer_.close()
        return qimage

    def drawPinholeMap(self ):
        sig = self.WidePinHole[0:300, 0, 0] - np.mean(self.WidePinHole[:, 0, 0])
        evp = signal.hilbert2(sig)
        envelope = np.abs(evp)
        # envelope = np.real( evp)
        print "evp max =", np.max(envelope), "arg max =", np.argmax(envelope)
        plt.close("all")
        print "drawCurrentPinhole"
        # plt.gray()
        # ax = plt.gca()
        z = plt.axvline(x=frameNo, linewidth=2, color='r')
        # ax.add_artist(z)
        plt.plot(envelope, color='r')
        plt.plot(-envelope, color='g')
        plt.plot(sig)
        # plt.plot(self.WidePinHole[:, 1, 0])
        plt.show()
        from cStringIO import StringIO
        buffer_ = StringIO()
        plt.savefig(buffer_, format="png")
        buffer_.seek(0)
        image = Image.open(buffer_)
        from PIL.ImageQt import ImageQt
        qimage = ImageQt(image)
        buffer_.close()
        return qimage


    def calculateZpeaksCentroid(self):
        """

        :param self:
        :return:
        """
        self.ZpeaksCentroidMax = np.zeros((self.frmCounter, self.FrameWidth, self.FrameHeight), dtype=float)
        ZpeaksSlicing = np.zeros((self.frmCounter, self.FrameWidth, self.FrameHeight), dtype=float)

        pass
