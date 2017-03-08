import os
import configparser

def getScanConfig(path):
    """
    Read Max-Inspect ini-file and create configparser object
    :param path: path to the ini file
    :return: A configparser object
    """
    config = configparser.ConfigParser()
    config.read(path)
    return config

def  iniAndSurfPath(surfPath):
    """
    Determines if surface and ini files exist and return both paths to them
    :param surfPath:  surface file path
    :return: both surface and ini file full paths
    """
    if not os.path.isfile(surfPath):
        return None, None
    # name = os.path.splitext(iniOrSurf)[0]
    iniPath = surfPath + ".dim"
    if not os.path.isfile(surfPath):
        surfPath = None
    if not os.path.isfile(iniPath):
        iniPath = surfPath + ".ini"
    return iniPath, surfPath

# test
path = "C:\_Projects\python\SurfaceInvestigation\Surface\surfaceExport0.ini"
config = getScanConfig(path)
print(config['Device']['WaveLength'])
