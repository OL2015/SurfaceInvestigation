""" MiraSimulator2
 AT the moment we suppose magnification = 1.0
Contains
========
MiraSimulator2

"""
import numpy as np
from scipy import optimize as op
from PIL import Image, ImageDraw
from collections import OrderedDict
import FiberHelpers as fh
import DeviceParams as dv
DEBUG = False
CONTRAST_BASE = 230

if DEBUG:
    import matplotlib.pyplot as plt


class MiraSimulator2(object):
    """A MiraSimulator.
       """
    Error = ""

    def __init__(self, devParams, **kwargs):
        self.DevParams = devParams
        self.FiberDiam = self.DevParams["FiberDiam"]
        self.UnderMaskDict = OrderedDict()
        self.SharpnessDict = OrderedDict()
        if DEBUG: print "MiraSimulator2.ctor():"
        if DEBUG: print "    fiber dia: ", self.FiberDiam, "um"
    pass


    def loadMiraImage(self, path, cropPadding = 20):
        """
        Loads mira image. Then crop it
        If not loaded, imgSize should be set
        :param path: - full path to the image of mira
        cropPadding: - crops image leaving a pad of cropPadding pixels around the fiber
        :return:
        """
        try:
            self.Results = OrderedDict()
            self.Path = path
            mira = Image.open(path, 'r')
            w, h = mira.size
            pix = np.array(mira)
            #reduce image size to search fiber
            pix = pix[h/2 - 500:h/2 + 500, w/2 - 500:w/2 + 500]  # crop image to center
            d = self.FiberDiam
            cx, cy = fh.findFiber(pix, d)
            r = int(d / 2)
            # crop image around the found fiber
            pixCropped = pix[cy - r - cropPadding: cy + r + cropPadding, cx - r - cropPadding: cx + r + cropPadding, ]  # crop image to center
            self.Mira = Image.fromarray(pixCropped.copy())
            # self.Mira.save("C:/temp/mira_cropped.bmp")
            self.imgSize = self.Mira.size
            self.FiberCenter = (self.imgSize[0] / 2 ,self.imgSize[1] / 2)
            self.setFiberCenter()
            # self.normalizeImage()
            return self.Mira
        except:
            self.Error = "Cannot load image"
            raise

    def measure(self):
        #
        try:
            self.fiberFerruleBright()

            self.calculateSharpness()
            self.normalizeImage()
            self.calculateFoucaultContrast()
        except:
            self.Error = "Calculation Error"
            raise

    def setFiberCenter(self):
        for k, v in self.DevParams['rois'].iteritems():
            if DEBUG: print "{0} = {1}".format(k, v)
            v.setFiberCenter (self.FiberCenter)


    def drawSharpnessROI(self):
        mask = Image.new('L', self.imgSize, color=0)
        draw = ImageDraw.Draw(mask)
        for k, v in self.DevParams['rois'].iteritems():
            print "{0} = {1}".format(k, v)
            draw.rectangle(v.getRoiRectangle(), fill=1, outline=1)
        np_mira = np.array(self.Mira)
        np_mask = np.array(mask)

        np_masked = np_mira
        np_masked[np_mask == 0] -= 35
        masked = Image.fromarray(np_masked)
        self.ROIImage = masked
        self.UnderMaskDict.clear()
        for k, v in self.DevParams['rois'].items():
            tpl = v.getRoiTuple()
            self.UnderMaskDict[k] = self.Mira.crop(tpl )
        if DEBUG :
            plt.gray()
            #
            fig4 = plt.figure( )
            fig4.subplots_adjust(hspace=0.05, wspace=0.05)
            ax1 = fig4.add_subplot(241)
            ax1.imshow(self.Mira, interpolation='none', vmin=0, vmax=255)
            ax1.set_title('Mira Image')
            # ax1.xaxis.set_visible(False)
            # ax1.yaxis.set_visible(False)
            ax2 = fig4.add_subplot(243)
            ax2.imshow(masked, interpolation='none', vmin=0, vmax=255)
            ax2.set_title('Mira ROIs')

            ax3 = fig4.add_subplot(245)
            ax3.imshow(underMask20V, interpolation='none', vmin=0, vmax=255)
            ax3.set_title('2.0 mc Vertical ROI')

            ax4 = fig4.add_subplot(246)
            ax4.imshow(underMask20H, interpolation='none', vmin=0, vmax=255)
            ax4.set_title('2.0 mc Horizontal ROI')

            ax5 = fig4.add_subplot(247)
            ax5.imshow(underMask10V, interpolation='none', vmin=0, vmax=255)
            ax5.set_title('1.0 mc Vertical ROI')

            ax6 = fig4.add_subplot(248)
            ax6.imshow(underMask10H, interpolation='none', vmin=0, vmax=255)
            ax6.set_title('1.0 mc Horizontal ROI')

            ax9 = fig4.add_subplot(247)
            ax9.imshow(underMask050V, interpolation='none', vmin=0, vmax=255)
            ax9.set_title('0.50 mc Vertical ROI')

            ax10 = fig4.add_subplot(248)
            ax10.imshow(underMask050H, interpolation='none', vmin=0, vmax=255)
            ax10.set_title('0.50 mc Horizontal ROI')

            plt.show()
        # fig4.show()

    def createROIImage (self, roi ):
        """
        Creates a mask for regions that contain mira line groums for different resolutions
        :param angleGrad:
        :return:
        """
        mask = Image.new('L', self.imgSize, color=0)
        draw = ImageDraw.Draw(mask)
        rct = roi.getRoiRectangle( )
        draw.rectangle(rct, fill=1, outline=1)
        np_mira = np.array(self.Mira)
        np_mask = np.array (mask )
        np_masked = np_mira.copy()
        np_masked[np_mask==0] -= 25
        masked = Image.fromarray(np_masked)
        tpl1V = roi.getRoiTuple( )
        return mask, masked.crop(tpl1V)

    def calculateSharpness(self):
        if DEBUG: print "Image {0}  ".format(self.Path)
        self.drawSharpnessROI()
        self.SharpnessDict.clear()
        for k, v in self.DevParams['rois'].iteritems():
            mask, underMask = self.createROIImage(v)
            sharpness, sharpnessX, sharpnessY = fh.imageSharpness(underMask)
            sharpFeatures = {'mask':  mask, 'underMask': underMask, "sharpness" :sharpness,  "gradientX" :sharpnessX, "gradientY" :sharpnessY }
            self.SharpnessDict [k] = sharpFeatures
            if DEBUG: print " {2} gradientX={0},  gradientY={1}, , ".format(sharpFeatures["gradientX"], sharpFeatures["gradientY"], k)

    def normalizeImage(self, fiberNorm = 75, ferrNorm = 205):
        fiberBr, ferruleBr = self.fiberFerruleBright()
        a = (ferrNorm -  fiberNorm )/ (ferruleBr -  fiberBr )
        b =  ferrNorm - a * ferruleBr
        if DEBUG: print "a={0}, b={1}".format (a, b)
        np_mira = np.array(self.Mira)
        np_miraNorm = np.array(np_mira, np.float32)
        np_miraNorm =np.minimum( np.maximum(  np_miraNorm * a + b,  0), 255)
        miraNorm = Image.fromarray(np.array(np_miraNorm, np.int8), mode='L')
        self.Mira = miraNorm

    def fiberFerruleBright(self):
        np_mira = np.array(self.Mira)
        fiberRing = self.getFiberRing()
        fiber_masked = np_mira.copy()
        fiber_masked[ fiberRing ]  = 0
        fiberRingImage = Image.fromarray(fiber_masked)
        if DEBUG :
            plt.gray()
            plt.imshow(fiberRingImage, interpolation='none')
            plt.show()
        connectorBr = np_mira.mean()
        # self.fiberBr = np_mira[fiberRing ].mean()
        self.Results["fiberBr"] = np_mira[fiberRing ].mean()
        ferruleRing = self.getFerruleRing()
        # self.ferruleBr = np_mira[ferruleRing ].mean()
        self.Results["ferruleBr"] = np_mira[ferruleRing].mean()
        # print  "FiberBrightness={0}, FerruleBrightness={1},  Connector = {2}".format(self.fiberBr, self.ferruleBr, connectorBr )
        if DEBUG: print  "FiberBrightness={0}, FerruleBrightness={1},  Connector = {2}".format(self.Results["fiberBr"], self.Results["ferruleBr"], connectorBr)
        return self.Results["fiberBr"], self.Results["ferruleBr"]
        pass

    def getFiberRing(self):
        RingThick = 20
        return self.getRing(self.FiberDiam - RingThick/2 , self.FiberDiam )

    def getFerruleRing(self):
        RingThick = 20
        return self.getRing(self.FiberDiam + RingThick  , self.FiberDiam + 3*RingThick/2 )

    def getRing(self, innerD, outD):
        outerMask = fh.getCircleMask(self.imgSize, self.FiberCenter[0], self.FiberCenter[1], outD)
        innerMask = fh.getCircleMask(self.imgSize, self.FiberCenter[0], self.FiberCenter[1], innerD)
        RingMask = outerMask & ~innerMask
        if DEBUG: print  "RingMask = "
        if DEBUG: print RingMask
        return RingMask

    def calculateFoucaultContrast(self):
        for k, v in self.SharpnessDict.iteritems():
            mask, underMask = v["mask"], v["underMask"] #self.createROIImage(self.DevParams["roi20H"])
            np_underMask  = np.array(underMask)
            ctH, ctV = self.calculateFoucaultContrastUnderMask(np_underMask, np_underMask, k)
            self.SharpnessDict[k]["ctH"] = ctH
            self.SharpnessDict[k]["ctV"] = ctV
            if k in self.DevParams['passfail'] :
                a = (ctH >= self.DevParams['passfail'][k] [0][0]) & (ctH <= self.DevParams['passfail'][k] [0][1])
                b = (ctV >= self.DevParams['passfail'][k][1][ 0]) & (ctV <= self.DevParams['passfail'][k][1][1])
                print self.DevParams['passfail'][k][1][ 0], self.DevParams['passfail'][k][1][1]
                self.SharpnessDict[k] ["verdictCT"]  = [a, b]

    def calculateFoucaultContrastUnderMask(self, np_underMaskH, np_underMaskV, titte = ""):
        np_averageH = np.average(np_underMaskH, 0)
        if DEBUG: print np_averageH
        # now do the fit
        t = range(np_averageH.size)
        ctH = (np.max(np_averageH) - np.min(np_averageH)) / CONTRAST_BASE
        # optimize_func = lambda  p : ctH * 255 * np.sin( p[1] * t +p[2] ) + p[3]   - np_averageH
        # guess_amplitude = 1; guess_freq = 1 ; guess_phase = 1; guess_offset = 130
        #
        # guess_amplitude, guess_freq, guess_phase, guess_offset = op.leastsq(optimize_func,  [guess_amplitude, guess_freq, guess_phase, guess_offset] )[0]
        # recreate the fitted curve using the optimized parameters
        #data_fit = optimize_func(   [guess_amplitude, guess_freq, guess_phase, guess_offset] )
        np_underMaskVT = np.transpose(np_underMaskV)
        np_averageVT = np.average(np_underMaskVT, 0)
        if DEBUG: print np_averageVT
        # now do the fit
        t = range(np_averageVT.size)
        ctV = (np.max(np_averageVT) - np.min(np_averageVT)) / 255

        # mask, und erMask = self.createROIImage(self.roi20V)
        # print  np.array (mask ), np.array (underMask)
        if DEBUG:
            plt.gray()
            #
            fig5 = plt.figure()
            fig5.suptitle(titte, fontsize=20)
            fig5.subplots_adjust(hspace=0.05, wspace=0.05)
            ax1 = fig5.add_subplot(211)
            ax1.set_title('Foucault H')
            for r in np_underMaskH[:]:
                ax1.plot(r)  # np_underMask[0, :]
            ax1.plot(np_averageH, linewidth=2.0, color='r')
            # ax1.plot(data_fit, linewidth=4.0, color='b')
            ax2 = fig5.add_subplot(212)
            ax2.set_title('Foucault V')
            for r in np_underMaskVT[:]:
                ax2.plot(r)  # np_underMask[0, :]
            ax2.plot(np_averageVT, linewidth=3.0, color='r')
            plt.show()
        return ctH, ctV
