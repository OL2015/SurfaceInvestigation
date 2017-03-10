__author__ = 'Oleksandr Shpylka'

import numpy as np
class Image2D(object):
    def __init__(self,W,H,R):
        

        self.W = W
        self.H = H
        self.R = R
        self.image = np.zeros((W,H), dtype = float)
    pass


    def ImageTwoBowl(self):
        x1 = np.arange(0, self.H / 2, 1) - self.H / 2 + self.R  # pix
        x2 = np.arange(0, self.H / 2, 1) - self.R + 1
        y = np.arange(0, self.W, 1) - self.W / 2  # pix
        xx1, yy = np.meshgrid(x1, y)  # pix
        xx2, yy = np.meshgrid(x2, y)  # pix
        xx = np.concatenate((xx1, xx2), axis=1)
        yy = np.concatenate((yy, yy), axis=1)
        self.image = np.sqrt((xx ** 2 + yy ** 2).astype(float))
        self.image = (self.image >= self.R).astype(float)
    pass

    def BlurFilter(self, N, sigma):
        from scipy import signal
        x = np.arange(1, 2 * N, 1) - N
        y = np.arange(1, 2 * N, 1) - N
        xx, yy = np.meshgrid(x, y)
        window = np.exp(-(xx ** 2 + yy ** 2) / 2 / sigma) / np.sqrt(2 * np.pi * sigma)
        window = window / np.sum(window)
        self.image = signal.convolve2d(self.image, window, mode='same')
    pass

    def AddGradient(self,A,B,D):
        x = np.arange(0, self.H,1)
        y = np.arange(0, self.W,1)
        xx, yy = np.meshgrid(x, y)
        Gradient = (A*xx + B*yy +D).astype(float);
        self.image = self.image - Gradient
    pass

    def AddGaussianNoise(self,sigma2):
        self.image = self.image + np.random.normal(loc=0., scale=sigma2, size=(self.H,self.W))
    pass

    def AddShadow(self,Yaw,Pitch):
        x1 = np.arange(0, self.H / 2, 1) - self.H / 2 + self.R  # pix
        x2 = np.arange(0, self.H / 2, 1) - self.R + 1
        y = np.arange(0, self.W, 1) - self.W / 2  # pix
        xx1, yy = np.meshgrid(x1, y)  # pix
        xx2, yy = np.meshgrid(x2, y)  # pix
        xx = np.concatenate((xx1, xx2), axis=1)
        yy = np.concatenate((yy, yy), axis=1)
        height = np.sqrt((xx ** 2 + yy ** 2).astype(float))
        height = np.sqrt((xx ** 2 + yy ** 2).astype(float))*(height <= self.R).astype(float) + self.R*(height > self.R).astype(float)
        height = np.sqrt(self.R**2 - height**2)
        shadow = height*np.sin(Pitch*np.pi/180.)
        #np.savetxt('test.txt',shadow,fmt='%4.2f')
#        self.image = height
    pass

    def SaveImage(self,name):
        import matplotlib.pyplot as plt
        plt.imsave(name,self.image,cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    pass    

pass






