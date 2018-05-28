#!/usr/bin/env python

import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
from os import environ
import root_numpy

NN_Mode='kart'

def pol2kar_x(norm, phi):
    x = []
    x = np.sin(phi[:])*norm[:]
    return(x)
def pol2kar_y(norm, phi):
    y = []
    y = np.cos(phi[:])*norm[:]
    return(y)

def kar2pol(x, y):
    rho = np.sqrt(np.multiply(x,x) + np.multiply(y,y))
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2kar(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

def loadData(fName):
    treeName = 't'
    arrayName = rnp.root2array(fName, branches=['Boson_Pt', 'Boson_Phi', 'NVertex' ,
        'recoilslimmedMETsPuppi_Pt', 'recoilslimmedMETsPuppi_Phi', 'recoilslimmedMETsPuppi_sumEt',
        'recoilslimmedMETs_Pt', 'recoilslimmedMETs_Phi', 'recoilslimmedMETs_sumEt',
        'recoilpatpfNoPUMET_Pt','recoilpatpfNoPUMET_Phi', 'recoilpatpfNoPUMET_sumEt',
        'recoilpatpfPUCorrectedMET_Pt', 'recoilpatpfPUCorrectedMET_Phi', 'recoilpatpfPUCorrectedMET_sumEt',
        'recoilpatpfPUMET_Pt', 'recoilpatpfPUMET_Phi', 'recoilpatpfPUMET_sumEt',
        'recoilpatpfTrackMET_Pt', 'recoilpatpfTrackMET_Phi', 'recoilpatpfTrackMET_sumEt' ],)
    DFName = pd.DataFrame.from_records(arrayName.view(np.recarray))
    return(DFName)

def prepareOutput(outputD, inputD):
    NN_Output = h5py.File("%sNN_Output.h5"%outputD, "r+")
    mZ_x, mZ_y = NN_Output['MET_GroundTruth'][:,0], NN_Output['MET_GroundTruth'][:,1]
    a_x, a_y = NN_Output['MET_Predictions'][:,0], NN_Output['MET_Predictions'][:,1]
    a_ =  np.sqrt(np.add(np.multiply(a_x,a_x),np.multiply(a_y,a_y)))
    #MET_x, MET_y
    a_r,a_phi = kar2pol(a_x,a_y)
    mZ_r,mZ_phi = kar2pol(mZ_x, mZ_y)
    NN_LongZ, NN_PerpZ= pol2kar(a_r,a_phi-mZ_phi)
    #NN_LongZ = div0(np.multiply(a_x, mZ_x)+np.multiply(a_y, mZ_y) , list(map(np.sqrt,np.add(np.multiply(mZ_x,mZ_x),np.multiply(mZ_y,mZ_y)))))
    #NN_PerpZ = np.sqrt(np.subtract(np.multiply(a_,a_), np.multiply(NN_LongZ,NN_LongZ)))
    dset = NN_MVA.create_dataset("NN_LongZ", dtype='f', data=NN_LongZ)
    dset = NN_MVA.create_dataset("NN_PerpZ", dtype='f', data=NN_PerpZ)
    NN_MVA.close()

    #branch_Long = np.array(NN_LongZ , dtype=[('NN_LongZ', 'f8')])
    branch_Long_Perp = np.array([NN_LongZ,NN_PerpZ] , dtype=[('NN_LongZ', 'f8'),(('NN_PerpZ', 'f8'))])
    root_numpy.array2root(branch_Long_Perp, inputD, tree='MAPAnalyzer/t' )
    #branch_Perp = np.array(NN_PerpZ , dtype=[('NN_PerpZ', 'f8')])
    #root_numpy.array2root(branch_Perp, inputD)









if __name__ == "__main__":
    inputDir = sys.argv[1]
    outputDir = sys.argv[2]
    print(outputDir)
    NN_MVA = h5py.File("%s/NN_MVA.h5"%outputDir, "w")
    prepareOutput(outputDir, inputDir)
