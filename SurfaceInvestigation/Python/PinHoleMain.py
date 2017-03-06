__author__ = 'sumix '
import os
from threading import Timer


import numpy as np
from scipy import misc


from PyQt4.uic import loadUiType
from PyQt4.QtGui import QFileDialog, QPixmap
from PyQt4.QtCore import Qt, QCoreApplication, QEvent
from PIL.ImageQt import ImageQt
from ctypes import *
import sys
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from SurfaceAnalizer import ScanParameters
from SurfaceAnalizer import PinHoleAnalizer
from  DeviceSimulator import DeviceParams as dp


#
# PassFailBr = 110
#
# mpp = 0.19
# precisionUm = 0.09
# maxShiftUm = 20
#
# modelMpp= mpp/10
# fiberDia = 125
# imgSize = 1.3*125
# fiberColor = 48
# ferruleColor = 142
# linesColor = 225
scale =  4.0
# devList = dp.getDevList()

Ui_MainWindow, QMainWindow = loadUiType(
os.path.dirname(os.path.realpath(__file__))+'\\PinHoleMain.ui')
defautPath = "C:\\_Projects\\python\\SurfaceInvestigation\\RawDataLastScan\\ScanRawData.dat"
noiseCalibrationPath = "C:\\_Projects\\python\\SurfaceInvestigation\\_PatternCoefs\\PC4547.dat"
outRemovedNoisePath = "C:\\_Projects\\python\\Surface\\RemovedNoise\\ScanRawData.dat"
class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, ):
        super(Main, self).__init__()
        self.setupUi(self)
        self.btnPath1.clicked.connect(self.selectFile1)
        self.btnLoadSurface.clicked.connect(self.loadSurface)
        self.viewer = self.findChild(QtGui.QGraphicsView, "gvSurfImage")
        print self.viewer
        self.viewer.setMouseTracking(True)
        # self.viewer.installEventFilter(self)
        self.scene = self.viewer.viewport()
        print self.scene
        self.scene.setMouseTracking(True)
        self.scene.installEventFilter(self)
        self.viewer.scale(scale, scale)
        self.surfacePath.setText(defautPath)
        #
        self.sliderFrame.valueChanged.connect(self.frameChanged)
        self.btnPlay.clicked.connect(self.play)
        self.btnStop.clicked.connect(self.stop)
        self.btnFrameBackward.clicked.connect(self.privFrame)
        self.btnFrameForvard.clicked.connect(self.nextFrame)
        self.btnFirstFrame.clicked.connect(self.firstFrame)
        self.btnLastFrame.clicked.connect(self.lastFrame)
        self.btnShowPinhole.clicked.connect(self.showPinhile )

    def eventFilter(self, source, event):
        if (  source is self.scene
                    # source is self.viewer.viewport()
                 and event.type() == QEvent.MouseButtonRelease
                 and  event.button() == Qt.LeftButton
            ):
            pos = event.pos()
            print ('x=%0.01f,y=%0.01f' % (pos.x(), pos.y()))
            point =  self.viewer.mapToScene(pos.x(), pos.y())
            print ('x=%0.01f,y=%0.01f' % (point.x(), point.y()))
            x = int(point.x()) ; y = int(point.y())
            self.x = x; self.y = y
            self.sbX.setValue(x)
            self.sbY.setValue(y)
            self.showPinhile()
        return QtGui.QWidget.eventFilter(self, source, event)

    def closeEvent(self, event):
        # do stuff
        event.accept()  # let the window close
        QCoreApplication.instance().quit()

    def selectFile1(self):
        d = os.path.dirname(os.path.abspath(str(self.surfacePath.text())))
        self.surfacePath.setText(QFileDialog.getOpenFileName( directory = d, filter ="*.dat" ))
        self.filePath = str(self.surfacePath.text())

    def loadSurface(self):
        self.filePath = str(self.surfacePath.text())
        self.dimpath, self.surfacepath = ScanParameters.iniAndSurfPath(self.filePath)

        print self.dimpath
        print self.surfacepath
        self.pha = PinHoleAnalizer.PinHoleAnalizer(self.dimpath, self.surfacepath)
        self.sliderFrame.setMinimum(0);  self.sliderFrame.setMaximum(self.pha.frmCounter-1)
        print self.pha.FrameWidth, self.pha.FrameHeight
        print self.pha.frmCounter
        self.lblTotalFrames.setText(str(self.pha.frmCounter))
        self.showFrame(0)
        self.sbX.setValue(self.pha.FrameWidth/2)
        self.sbY.setValue(self.pha.FrameHeight/2)
        # QtCore.QTimer.singleShot(0, self.updateScreen)
        # QtCore.QTimer.singleShot(1000, self.jubba)
        # self.animator = QtCore.QTimer()
        # self.animator.timeout.connect(self.animate)
        # self.animate()
        # self.pha.removePatternNoise(noiseCalibrationPath, outRemovedNoisePath)
        self.pha.readPiezoDistanse()

    def showPinhile(self,  ):
        x = self.sbX.value()
        y= self.sbY.value()
        fn = self.sliderFrame.value()
        if (x!=None and y!=None):
            pinhole = self.pha.sliceRawData(x, y, 1, 1)
        qimage = self.pha.drawCurrentPinhole(fn)
        viewer = self.findChild(QtGui.QGraphicsView, "gvPinHole")
        scene = QtGui.QGraphicsScene(viewer)
        # scene.addPixmap(QtGui.QPixmap.fromImage(image))
        pixmap = QtGui.QPixmap.fromImage(qimage)
        scene.addPixmap(pixmap)
        pen = QtGui.QPen(Qt.red)
        point =  viewer.mapToScene( x  , y )
        viewer.setScene(scene)
        scene.addEllipse(point.x(),point.y(), 3,3, pen)





    def showFrame(self, n ):
        img, qhist = self.pha.getFrame(n)
        viewer = self.findChild(QtGui.QGraphicsView, "gvSurfImage")
        image = QtGui.QImage(np.array(img), img.size[0], img.size[1], QtGui.QImage.Format_Indexed8)
        viewer.scene = QtGui.QGraphicsScene(viewer)
        viewer.scene.addPixmap(QtGui.QPixmap.fromImage(image))
        viewer.setScene(viewer.scene)
        self.lblCurrentFrame.setText(str(n))

        histoViewer = self.findChild(QtGui.QGraphicsView, "gvHisto")
        pixmap = QtGui.QPixmap.fromImage(qhist)
        histoViewer.scene = QtGui.QGraphicsScene(histoViewer)
        histoViewer.scene.addPixmap(pixmap)

        histoViewer.setScene(histoViewer.scene)
        pass

    def frameChanged(self):
        v = self.sliderFrame.value()
        self.showFrame(v)
        self.showPinhile(self.x, self.y)

    def privFrame(self ):
        v = self.sliderFrame.value()
        if v>0:
            v=v-1
            self.sliderFrame.setValue(v)
            # self.showFrame(v)
    def nextFrame(self):
        v = self.sliderFrame.value()
        if v < self.pha.frmCounter -1:
            v += 1
            self.sliderFrame.setValue(v)
            # self.showFrame(v)
    def firstFrame(self):
        self.sliderFrame.setValue(0)
    def lastFrame(self):
        self.sliderFrame.setValue(self.pha.frmCounter -1)

    def wheelEvent(self, event):
        viewer = self.findChild(QtGui.QGraphicsView, "gvSurfImage")
        delta = event.delta()
        if delta > 100:
            viewer.scale(1.1, 1.1)
        else:
            viewer.scale(0.9, 0.9)
        print delta

    # def timeout(self):
    #     v =self.sliderFrame.value()
    #     print v
    #     if v < self.sliderFrame.maximum():
    #         self.sliderFrame.setValue( self.sliderFrame.value() +1)
    #     else :
    #         self.sliderFrame.setValue(0)
    #     self.t = Timer(2, self.timeout)

    def play(self):
        self.t.start()

    def stop(self):
        self.t.stop()

if __name__ == '__main__':
    import sys
    from PyQt4 import QtGui

    app = QtGui.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())