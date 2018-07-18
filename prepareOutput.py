#!/usr/bin/env python

import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import root_numpy as rnp
import sys
import pandas as pd
import numpy as np
from os import environ
import root_numpy
from rootpy.tree import Tree, TreeModel, FloatCol, IntCol
from rootpy.io import root_open
from prepareInput import pol2kar_x, pol2kar_y, kar2pol, pol2kar, angularrange
import ROOT



def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

def loadData(fName, NN_mode):
    treeName = 't'
    arrayName = rnp.root2array(fName, branches=[Target_Pt,
        'recoilslimmedMETs_Pt', Target_Phi],)
    DFName = pd.DataFrame.from_records(arrayName.view(np.recarray))
    return(DFName)

def prepareOutput(outputD, inputD, NN_mode, plotsD):
    NN_Output = h5py.File("%sNN_Output_applied_%s.h5"%(outputD,NN_mode), "r+")
    if NN_mode == 'xyr' or NN_mode == 'nr':
        mZ_x, mZ_y, mZ_r = NN_Output['MET_GroundTruth'][:,0], NN_Output['MET_GroundTruth'][:,1], NN_Output['MET_GroundTruth'][:,2]
        a_x, a_y, a_r = NN_Output['MET_Predictions'][:,0], NN_Output['MET_Predictions'][:,1], NN_Output['MET_Predictions'][:,2]
    elif NN_mode == 'xyd':
        PF_Z_pT = loadData(inputD, NN_mode)
        mZ_x, mZ_y, mZ_r = NN_Output['MET_GroundTruth'][:,0], NN_Output['MET_GroundTruth'][:,1], PF_Z_pT[Target_Pt]
        a_x, a_y, a_r = -NN_Output['MET_Predictions'][:,0], -NN_Output['MET_Predictions'][:,1], NN_Output['MET_Predictions'][:,2]+PF_Z_pT['recoilslimmedMETs_Pt']
    elif NN_mode == 'rphi':
        mZ_r, mZ_phi = NN_Output['MET_GroundTruth'][:,0], NN_Output['MET_GroundTruth'][:,1]
        a_r, a_phi = NN_Output['MET_Predictions'][:,0], (NN_Output['MET_Predictions'][:,1])
        a_x, a_y = kar2pol(a_r, a_phi)
        mZ_r, mZ_phi =  pol2kar(mZ_r, mZ_phi)
    elif NN_mode == 'xy':

        mZ_x, mZ_y = (NN_Output['MET_GroundTruth'][:,0]), (NN_Output['MET_GroundTruth'][:,1])
        a_x, a_y = (NN_Output['MET_Predictions'][:,0]), (NN_Output['MET_Predictions'][:,1])
        mZ_r, mZ_phi =  kar2pol(mZ_x, mZ_y)
        mZ_r = NN_Output['Boson_Pt'][:]
        a_r, a_phi = np.sqrt(a_x*a_x+a_y*a_y), np.arctan2(a_y, a_x)
    else:
        mZ_x, mZ_y = NN_Output['MET_GroundTruth'][:,0], NN_Output['MET_GroundTruth'][:,1]
        a_x, a_y = NN_Output['MET_Predictions'][:,0], (NN_Output['MET_Predictions'][:,1])
        mZ_r, mZ_phi =  kar2pol(mZ_x, mZ_y)

    a_ =  np.sqrt(np.add(np.multiply(a_x,a_x),np.multiply(a_y,a_y)))
    #r_ =  np.sqrt(np.add(np.multiply(mZ_x,mZ_x),np.multiply(a_y,a_y)))
    #MET_x, MET_y
    if NN_mode=="xyr" or NN_mode=="nr" or NN_mode=="xyd":
        Scale = np.sqrt(div0(np.multiply(a_r,a_r), np.multiply(a_,a_)))
        a_x = a_x*Scale
        a_y = a_y*Scale
        print('mZ_r',mZ_r)
        Scale_Z = np.sqrt(mZ_r)
        mZ_x = mZ_x*Scale_Z
        mZ_y = mZ_y*Scale_Z
    mZ_r2,mZ_phi = kar2pol(mZ_x, mZ_y)
    print("mZ_r,mZ_phi", mZ_r,mZ_phi)
    #a_r,a_phi = kar2pol(a_x,a_y)
    a_x2, a_y2 = pol2kar(a_r, a_phi)
    print('Diff Test a_x, a_y', a_x2-a_x, a_y2-a_y)
    print("a_r,a_phi", a_r,a_phi)
    print('a_-a_r', a_-a_r)

    #Diff_phi = np.arccos(np.divide(np.add(np.multiply(a_x, mZ_x), np.multiply(a_y, mZ_y)), np.multiply(a_, mZ_r)))
    #NN_LongZ = np.divide(np.add(np.multiply(a_x, mZ_x), np.multiply(a_y, mZ_y)), mZ_r)
    #NN_PerpZ = np.sin(Diff_phi)*a_

    #NN_LongZ = div0(np.add(np.multiply(a_x, mZ_x) , np.multiply(a_y, mZ_y)), mZ_r)
    #ParaVx, ParaVy = div0(NN_LongZ*mZ_x, mZ_r), div0(NN_LongZ*mZ_y, mZ_r)
    #NN_PerpZ = np.sqrt( np.multiply(a_x-ParaVx, a_x-ParaVx) + np.multiply(a_y-ParaVy, a_y-ParaVy) )
    NN_LongZ, NN_PerpZ = -np.cos(angularrange(np.add(a_phi,-mZ_phi)))*a_, np.sin(angularrange(a_phi-mZ_phi))*a_
    print('np.isnan(NN_LongZ)', sum(np.isnan(NN_LongZ)))
    print('np.isnan(NN_PerpZ)', sum(np.isnan(NN_PerpZ)))
    print('np.isnan(a_phi)', sum(np.isnan(a_phi)))
    print('np.isnan(a_r)', sum(np.isnan(a_r)))
    #NN_LongZ, NN_PerpZ= pol2kar(a_r,angularrange(a_phi-mZ_phi))
    #NN_PerpZ[angularrange(a_phi-mZ_phi)<0]= -NN_PerpZ[angularrange(a_phi-mZ_phi)<0]
    #NN_LongZ = -NN_LongZ
    dset = NN_MVA.create_dataset("NN_LongZ", dtype='d', data=NN_LongZ)
    dset1 = NN_MVA.create_dataset("NN_PerpZ", dtype='d', data=NN_PerpZ)
    dset2 = NN_MVA.create_dataset("NN_Phi", dtype='d', data=a_phi)
    dset3 = NN_MVA.create_dataset("NN_Pt", dtype='d', data=a_r)
    dset4 = NN_MVA.create_dataset("Boson_Pt", dtype='d', data=mZ_r)
    dset5 = NN_MVA.create_dataset("NN_x", dtype='d', data=a_x)
    dset6 = NN_MVA.create_dataset("NN_y", dtype='d', data=a_y)
    dset7 = NN_MVA.create_dataset("Boson_x", dtype='d', data=mZ_x)
    dset8 = NN_MVA.create_dataset("Boson_y", dtype='d', data=mZ_y)
    NN_MVA.close()

    print('richtig, wenn auf a trainiert: -LongZ-pTZ', -NN_LongZ-mZ_r)
    print('richtig, wenn auf Z trainiert: LongZ-pTZ', np.add(NN_LongZ,-mZ_r))

    plt.clf()
    plt.figure()
    plt.suptitle('y: Prediction vs. Target ')
    plt.xlabel("$p_{T,y}^Z$")
    plt.hist(a_y, bins=50, range=[np.percentile(a_y,5), np.percentile( a_y,95)], histtype='step' )
    plt.hist(mZ_y, bins=50, range=[np.percentile(mZ_y,5), np.percentile( mZ_y,95)], histtype='step' )
    plt.legend(["Prediction","Target"], loc='upper left')
    plt.savefig("%sHist_Pred_Tar_y.png"%(plotsD))

    plt.clf()
    plt.figure()
    plt.suptitle('x: Prediction vs. Target ')
    plt.xlabel("$p_{T,x}^Z$")
    plt.hist(a_x, bins=50, range=[np.percentile(a_x,5), np.percentile( a_x,95)], histtype='step' )
    plt.hist(mZ_x, bins=50, range=[np.percentile(mZ_x,5), np.percentile( mZ_x,95)], histtype='step' )
    plt.legend(["Prediction","Target"], loc='upper left')
    plt.savefig("%sHist_Pred_Tar_x.png"%(plotsD))


    print('Mean x deviation', np.mean(np.subtract(a_x, mZ_x)))
    print('Std x deviation', np.std(np.subtract(a_x, mZ_x)))
    print('Mean y deviation', np.mean(np.subtract(a_y, mZ_y)))
    print('Std y deviation', np.std(np.subtract(a_y, mZ_y)))

    print('Summe a_x enspricht prediction 0', np.sum(-a_x))
    print('Summe a_y enspricht prediction 0', np.sum(-a_y))
    print('Summe prediction 0', np.sum(NN_Output['MET_Predictions'][:,0]))
    print('Summe isnan prediction 0', np.sum(np.isnan(NN_Output['MET_Predictions'][:,0])))
    print('Summe prediction 1', np.sum(NN_Output['MET_Predictions'][:,1]))
    print('Summe isnan prediction 1', np.sum(np.isnan(NN_Output['MET_Predictions'][:,1])))



if __name__ == "__main__":
    inputDir = sys.argv[1]
    outputDir = sys.argv[2]
    NN_mode = sys.argv[3]
    plotsD = sys.argv[4]
    PhysicsProcess = sys.argv[5]
    Target_Pt = 'Boson_Pt'
    Target_Phi = 'Boson_Phi'
    print(outputDir)
    NN_MVA = h5py.File("%s/NN_MVA_%s.h5"%(outputDir,NN_mode), "w")
    prepareOutput(outputDir, inputDir, NN_mode, plotsD)
