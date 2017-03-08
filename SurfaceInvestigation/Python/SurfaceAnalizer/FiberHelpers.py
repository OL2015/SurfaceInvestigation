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

def getRotationAngle(img1, img2, matchingDia):
    step = 1.0
    mask = getCircleMask(img1.shape, img1.shape[1]/2, img1.shape[0]/2,  matchingDia-2)
    avg = np.average(img1[mask])
    img1 = (1/avg)*img1.astype(np.float)
    avg = np.average(img2[mask])
    img2 = (1/avg)*img2.astype(np.float)

    roughAngle = FindOptAngle(img1, img2, mask, np.arange(0,360, step))
    preciseAngle = FindOptAngle(img1, img2, mask, np.arange(roughAngle-1,roughAngle+1, 0.01))

    #show results
    if False:
        rotatedImg = interp.rotate(img1, preciseAngle, reshape=False)
        imshow(rotatedImg, cmap="gray")
        show()

        imshow(img2, cmap="gray")
        show()
    return preciseAngle

#scratches are marked with red
#defects are marked with blue
def loadTemplate(pathToTemplate):
    img = misc.imread(pathToTemplate)
    #    pic = Image.open(filepath)
    # pix = np.array(pic)
    #
    # it is considered mask vals are marked with red
    scratchMask = (img[:, :, 0] != 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)
    defectMask = (img[:, :, 2] != 0) & (img[:, :, 1] == 0) & (img[:, :, 0] == 0)
    scratchIds = 256-img[:, :, 0]
    scratchIds[~scratchMask] = 0
    defectIds = -256+img[:, :, 2]
    defectIds[~defectMask] = 0

    return defectIds+scratchIds

def getRotatedMask(anomaliesMask, templateImage, testedImage, diaPx):
    angle = getRotationAngle(templateImage, testedImage, 0.8*diaPx)
    print "angle=", angle
    anomaliesMaskRotated = interp.rotate(anomaliesMask, angle, reshape=False, prefilter=False)
    templateImageRotated = interp.rotate(templateImage, angle, reshape=False)
    return anomaliesMaskRotated, templateImageRotated
    
def myHoughTransform(img, theta_bins=180, rho_bins=100):
    nR,nC = img.shape
  
    thetaRange = np.arange(0, 180, 180.0/theta_bins) 
    rho_max = np.sqrt(nR*nR+nC*nC)/2
    rho_step = 2.0*rho_max/rho_bins
    rhoRange = np.arange(-rho_max, rho_max, rho_step) 

    H = np.zeros((len(rhoRange), len(thetaRange)))        
    H_cnt = np.zeros((len(rhoRange), len(thetaRange)))        
    x0 = nC/2
    y0 = nR/2
    for rowIdx in range(nR):
        for colIdx in range(nC):
            if (~np.isnan(img[rowIdx, colIdx]) and img[rowIdx, colIdx]!=0):
                for thIdx in range(len(thetaRange)):
                    theta = thetaRange[thIdx]
                    x = colIdx-x0
                    y = rowIdx-y0
                    rhoVal = x * np.cos(theta * np.pi/180) + y * np.sin(theta * np.pi/180)
                    rhoIdx = int(rhoVal/rho_step+rho_bins/2)
                    H[rhoIdx, thIdx] += img[rowIdx, colIdx] 
                    H_cnt[rhoIdx, thIdx] += 1 
    #print len(thetaRange)*nR*nC," > ", np.sum(H_cnt)
    #H_cnt[H_cnt==0]=1
    return H, H_cnt, rhoRange, thetaRange, H_cnt

    
def getLocalPeaks(H, H_cnt, rhoRange, thetaRange, thresh, minPntCnt, normalize):
    vals = []
    rhos = []
    thetas = []
    rs = []
    cs = []
    if normalize:
        H_cnt_copy = np.copy(H_cnt)
        H_cnt_copy[H_cnt_copy==0]=1
        H_normed = H/H_cnt_copy
    else:
        H_normed = np.copy(H)
    for r in range(H_normed.shape[0]):
        for c in range(H_normed.shape[1]):
            if (H_cnt[r,c]<minPntCnt):
                continue
            #if is local max
            if (H_normed[r,c]>H_normed[r-1,c] and 
                H_normed[r,c]>H_normed[(r+1) % H_normed.shape[0],c] and 
                H_normed[r,c]>H_normed[r,c-1] and 
                H_normed[r,c]>H_normed[r,(c+1) % H.shape[1]]):
                vals.append(H_normed[r,c])
                rhos.append(rhoRange[r])
                thetas.append(thetaRange[c])
                rs.append(r)
                cs.append(c)
    avg = np.mean(H_normed)
    #print avg
    opts = [t for t in zip(vals, rhos, thetas, rs, cs) if t[0]/avg>thresh] 
    opts.sort(key=lambda t:t[0], reverse=True)
    return [t[0] for t in opts], [t[1] for t in opts], [t[2] for t in opts], [t[3] for t in opts], [t[4] for t in opts]
    
    

def getLocalPeaks2(H, rhoRange, thetaRange, thresh):
    vals = []
    rhos = []
    thetas = []
    rs = []
    cs = []
    for r in range(H.shape[0]):
        for c in range(H.shape[1]):
            if (H[r,c]>H[r-1,c] and H[r,c]>H[(r+1) % H.shape[0],c] and H[r,c]>H[r,c-1] and H[r,c]>H[r,(c+1) % H.shape[1]]):
                vals.append(H[r,c])
                rhos.append(rhoRange[r])
                thetas.append(thetaRange[c])
                rs.append(r)
                cs.append(c)
    
    opts = [t for t in zip(vals, rhos, thetas, rs, cs) if t[0]>thresh] 
    opts.sort(key=lambda t:t[0], reverse=True)
    return [t[0] for t in opts], [t[1] for t in opts], [t[2] for t in opts], [t[3] for t in opts], [t[4] for t in opts]

def getHoughSegments(shape, thetas, rhos):
    lines=[]
    rows, cols = shape
    for angle, dist in zip(thetas, rhos):
        angleRad = angle * np.pi/180
        if(angle >= 45 and angle <= 135):
            x1 = 0
            y1 = (dist - ((x1 - (cols/2) ) * np.cos(angleRad))) / np.sin(angleRad) + (rows / 2)
            x2 = rows - 1
            y2 = (dist - ((x2 - (cols/2) ) * np.cos(angleRad))) / np.sin(angleRad) + (rows / 2)
        else:
            y1 = 0
            x1 = (dist - ((y1 - (rows/2)) * np.sin(angleRad))) / np.cos(angleRad) + (cols / 2)
            y2 = cols - 1
            x2 = (dist - ((y2 - (rows/2)) * np.sin(angleRad))) / np.cos(angleRad) + (cols / 2)
        lines.append((x1,y1,x2,y2))
    return lines
    
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