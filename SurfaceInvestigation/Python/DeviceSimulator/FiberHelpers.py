__author__ = 'Vitalina'
import numpy as np
from ctypes import *
import os
from scipy.ndimage import interpolation as interp
from scipy import misc

DEBUG = False
def getCircleMask(shape, centerX, centerY, diaPx):
    y,x = np.ogrid[0:shape[0], 0:shape[1]]
    fiberMask = (x-centerX)**2+(y-centerY)**2 <= (diaPx/2)**2
    return fiberMask

def findFiber(img, dia):
    myPath = os.path.dirname(os.path.abspath(__file__))
    wd = os.getcwd()
    # try:
    if True:
        os.chdir(myPath+"/CLibsOld")
        dirPath = myPath+"/CLibsOld/CFiberDetector.dll"

        lib = WinDLL(dirPath)
        if DEBUG :
            print img.shape
        rows = img.shape[0]
        cols = img.shape[1]
        data = img.flatten().astype(np.uint8)

        data_p = data.ctypes.data_as(POINTER(c_uint8))
        c_x = c_double()
        c_y = c_double()
        c_dia = c_double(dia)

        lib.FindFiberCenter(c_dia, rows, cols, data_p, byref(c_x), byref(c_y))
    
        os.chdir(wd)
    # except:
    #     os.chdir(wd)
    return c_x.value, c_y.value

import matplotlib.pyplot as plt
def FindOptAngle(img1, img2, mask, angleRange):
    bestFitError = -1
    optAngle = -1
    for rotationAngle in angleRange:
        rotatedImg = interp.rotate(img1, rotationAngle, reshape=False)
        difference = rotatedImg-img2

        if False:
            d = np.abs(difference)
            d[~mask] = 0
            plt.imshow(d)
            plt.show()

        fitError = np.sum((difference**2)[mask])
        #print rotationAngle, fitError
        if (bestFitError<0 or fitError<bestFitError):
            bestFitError = fitError
            optAngle = rotationAngle
    return optAngle

class FastLineApproximator:
    def __init__(self):
        self.mat = np.zeros(3)
        self.col = np.zeros(2)
        
    def clear(self):
        self.mat[:] = 0
        self.col[:] = 0
        
    def addPoint(self, x, y, weight=1):
        self.mat[0]+=weight*x*x # corresponds to mat[0][0]
        self.mat[1]+=weight*x # corresponds to mat[0][1] and mat[1][0]
        self.mat[2]+=weight*1 # corresponds to mat[1][1]

        self.col[0]+=weight*x*y
        self.col[1]+=weight*y
        
    def pointsCnt(self):
        return int(self.mat[2]+0.5)
    
    def getResult(self):
        det = self.mat[0]*self.mat[2]-self.mat[1]*self.mat[1]
        detInv = 1.0/det
        coeff = detInv*(self.mat[2]*self.col[0]-self.mat[1]*self.col[1])
        shift = detInv*(-self.mat[1]*self.col[0]+self.mat[0]*self.col[1])
        return (coeff, shift)
    
def RemoveChessboard(img, threshold = 10, visualize = False):
    statistics = np.zeros(4*(4*256)*256)#(4 directions) x (4x256 avg values) x (256 values)
    vals =np.zeros(4);

    #Collect statistics
    for i in range(4):
        ir = i/2 #initial row
        ic = i%2 #initial col
        for row in range(ir, img.shape[0]-2, 2):
            for col in range(ic, img.shape[1]-2, 2):
                sumVal = 0
                skip = False
                for direc in range(4):
                    dr = direc/2
                    dc = direc%2
                    vals[direc] = img[row+dr, col+dc]
                    if vals[direc]==0 or vals[direc]==255:
                        skip = True; 
                        break;
                    sumVal += vals[direc]
                if skip:
                    continue;
                #gather statistics
                for direc in range(4):
                    r = (direc/2+ir)%2
                    c = (direc%2+ic)%2
                    pxlClass = 2*r+c
                    if abs(sumVal/4.0-vals[direc])<threshold:
                        statistics[(4*256)*256*pxlClass+256*sumVal+vals[direc]]+=1
    #Approximate stat data
    lineAprox = FastLineApproximator()
    approxResults = np.zeros((4,2))
    
    
    for direc in range(4):
        lineAprox.clear()
        if visualize:
            pntsX = []
            pntsY = []
            ws = []
        for realVal in range(4*256):
            cnt =0;
            for obsVal in range(256):
                num = statistics[(4*256)*256*direc+256*realVal+obsVal]
                if num>0:
                    lineAprox.addPoint(obsVal, realVal, num);
                    cnt+=1
                    if visualize:
                        pntsX.append(obsVal)
                        pntsY.append(realVal/4.0)
                        ws.append(10*num)
        if lineAprox.pointsCnt()<2:
            approxResults[direc][0] = 4
            approxResults[direc][1] = 0
        else:
            approxResults[direc,:] = lineAprox.getResult()
        if visualize:
            plt.figure(figsize=(15,5))
            plt.scatter(pntsX, pntsY, ws)
            plt.plot([0, 255], [approxResults[direc,1]/4, (approxResults[direc,0]*255+approxResults[direc,1])/4])
            plt.show()
    
    #evaluate minimal dynamic range
    blackMax = 0
    whiteMin = 255
    for direc in range(4):
        blackMax = max(blackMax, approxResults[direc][1]/4)
        whiteMin = min(whiteMin, (approxResults[direc][0]*255+approxResults[direc][1])/4)

    resImg = np.zeros(img.shape)
    for row in range(0, img.shape[0]-1, 2):
        for col in range(0, img.shape[1]-1, 2):
            for direc in range(4):
                dr = direc/2
                dc = direc%2
                observ = img[row+dr, col+dc]
                val = (approxResults[direc][0]*observ+approxResults[direc][1])/4
                val = min(whiteMin, max(blackMax, val))
                resImg[row+dr, col+dc] = int(val+0.5)
    return resImg
    
# OL added - 26.08.2016
def imageSharpness(img):
    """

    :param img:
    :return:
    """
    pxs =  np.array(img)
    pxs = np.transpose(pxs)
    gy, gx = np.gradient(pxs, 2)
    gnorm = np.sqrt(gx ** 2 + gy ** 2)
    sharpness = np.average(gnorm)
    from numpy import linalg
    pxs0 =np.array (  pxs[0: pxs.shape[0] - 2, :], np.int32)
    pxs1 =np.array (   pxs[1: pxs.shape[0] - 1, :], np.int32)
    a = pxs0 - pxs1
    pxs0 = np.array(pxs[:, 0: pxs.shape[1] - 2], np.int32)
    pxs1 = np.array(pxs[:, 1: pxs.shape[1] - 1], np.int32)
    b = pxs0 - pxs1
    # b = pxs[:, 0: pxs.shape[1] - 2] - pxs[:, 1: pxs.shape[1] - 1]
    if DEBUG:
        print "pxs.shape=", pxs.shape
        print "pxs=", pxs
        print "pxs0=", pxs0
        print "pxs1=", pxs1
        print "a=" , a
        print "b=", b
    sharpnessY = np.average( np.sqrt( np.sum(a*a, axis=0)))   / (pxs.shape[1] -1)
    sharpnessX = np.average( np.sqrt( np.sum(b*b, axis=1)))   / (pxs.shape[0]-1)

    return sharpness, sharpnessX, sharpnessY

def rotateImage(image, angle, **kwargs):
    # pinpoint- point to rotate around
  if ('pinpoint' in kwargs):
      center = kwargs['pinpoint']
  center = tuple(np.array(image.shape)/2)
  rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape, flags=cv2.INTER_LINEAR)
  return result

def getCircleMask( shape, centerX, centerY, diaPx):
    """
        Examples
        ========
    """
    y, x = np.ogrid[0:shape[0], 0:shape[1]]
    fiberMask = (x - centerX) ** 2 + (y - centerY) ** 2 <= (diaPx / 2) ** 2
    return  fiberMask