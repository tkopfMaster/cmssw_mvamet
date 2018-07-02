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
    arrayName = rnp.root2array(fName, branches=['Boson_Pt',
        'recoilslimmedMETs_Pt'],)
    DFName = pd.DataFrame.from_records(arrayName.view(np.recarray))
    return(DFName)

def prepareOutput(outputD, inputD, NN_mode, plotsD):
    NN_Output = h5py.File("%sNN_Output_applied_%s.h5"%(outputD,NN_mode), "r+")
    if NN_mode == 'xyr' or NN_mode == 'nr':
        mZ_x, mZ_y, mZ_r = NN_Output['MET_GroundTruth'][:,0], NN_Output['MET_GroundTruth'][:,1], NN_Output['MET_GroundTruth'][:,2]
        a_x, a_y, a_r = NN_Output['MET_Predictions'][:,0], NN_Output['MET_Predictions'][:,1], NN_Output['MET_Predictions'][:,2]
    elif NN_mode == 'xyd':
        PF_Z_pT = loadData(inputD, NN_mode)
        mZ_x, mZ_y, mZ_r = NN_Output['MET_GroundTruth'][:,0], NN_Output['MET_GroundTruth'][:,1], PF_Z_pT['Boson_Pt']
        a_x, a_y, a_r = -NN_Output['MET_Predictions'][:,0], -NN_Output['MET_Predictions'][:,1], NN_Output['MET_Predictions'][:,2]+PF_Z_pT['recoilslimmedMETs_Pt']
    elif NN_mode == 'rphi':
        mZ_r, mZ_phi = NN_Output['MET_GroundTruth'][:,0], NN_Output['MET_GroundTruth'][:,1]
        a_r, a_phi = NN_Output['MET_Predictions'][:,0], (NN_Output['MET_Predictions'][:,1])
        a_x, a_y = kar2pol(a_r, a_phi)
        mZ_r, mZ_phi =  pol2kar(mZ_r, mZ_phi)
    elif NN_mode == 'xy':
        mZ_x, mZ_y = (NN_Output['MET_GroundTruth'][:,0]), (NN_Output['MET_GroundTruth'][:,1])
        a_x, a_y = (NN_Output['MET_Predictions'][:,0])/0.42, (NN_Output['MET_Predictions'][:,1])/0.42
        mZ_r, mZ_phi =  kar2pol(mZ_x, mZ_y)
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
    a_r,a_phi = kar2pol(a_x,a_y)
    a_x2, a_y2 = pol2kar(a_r, a_phi)
    print('Diff Test a_x, a_y', a_x2-a_x, a_y2-a_y)
    print("a_r,a_phi", a_r,a_phi)
    print('a_-a_r', a_-a_r)

    Diff_phi = np.arccos(np.divide(np.add(np.multiply(a_x, mZ_x), np.multiply(a_y, mZ_y)), np.multiply(a_, mZ_r)))
    #NN_LongZ = np.divide(np.add(np.multiply(a_x, mZ_x), np.multiply(a_y, mZ_y)), mZ_r)
    #NN_PerpZ = np.sin(Diff_phi)*a_

    NN_LongZ = div0(np.add(np.multiply(a_x, mZ_x) , np.multiply(a_y, mZ_y)), mZ_r)
    ParaVx, ParaVy = div0(NN_LongZ*mZ_x, mZ_r), div0(NN_LongZ*mZ_y, mZ_r)
    NN_PerpZ = np.sqrt( np.multiply(a_x-ParaVx, a_x-ParaVx) + np.multiply(a_y-ParaVy, a_y-ParaVy) )
    #NN_LongZ, NN_PerpZ = -np.cos(angularrange(np.add(a_phi,-mZ_phi)))*a_, np.sin(angularrange(a_phi-mZ_phi))*a_
    #NN_LongZ, NN_PerpZ= pol2kar(a_r,angularrange(a_phi-mZ_phi))
    NN_PerpZ[angularrange(a_phi-mZ_phi)<0]= -NN_PerpZ[angularrange(a_phi-mZ_phi)<0]
    NN_LongZ = -NN_LongZ

    #
    #NN_LongZ, NN_PerpZ = NN_LongZ_l.tolist(), NN_PerpZ_l.tolist()
    #print("NN_LongZ, NN_PerpZ", NN_LongZ, NN_PerpZ)
    #NN_LongZ = div0(np.multiply(a_x, mZ_x)+np.multiply(a_y, mZ_y) , list(map(np.sqrt,np.add(np.multiply(mZ_x,mZ_x),np.multiply(mZ_y,mZ_y)))))
    #NN_PerpZ = np.sqrt(np.subtract(np.multiply(a_,a_), np.multiply(NN_LongZ,NN_LongZ)))
    dset = NN_MVA.create_dataset("NN_LongZ", dtype='d', data=NN_LongZ)
    dset = NN_MVA.create_dataset("NN_PerpZ", dtype='d', data=NN_PerpZ)
    NN_MVA.close()

    #branch_Long = np.array(NN_LongZ , dtype=[('NN_LongZ', 'f8')])
    branch_Long_Perp = np.array( zip(NN_LongZ,NN_PerpZ), dtype=[('NN_LongZ', 'f8'), ('NN_PerpZ', 'f8')])
    branch_Long = np.array(NN_LongZ , dtype=[('NN_LongZ', 'f8')])
    print("NN_LongZ, NN_PerpZ, numpy arrays", NN_LongZ[0:10], NN_PerpZ[0:10])
    print("branch_Long_Perp, numpy ndarrays", branch_Long_Perp[0:10])
    root_numpy.array2root(branch_Long, inputD, treename='t' , mode='update')
    #root_numpy.array2root(branch_Long_Perp, inputD, treename='t' , mode='update')
    branch_Perp = np.array(NN_PerpZ , dtype=[('NN_PerpZ', 'f8')])
    root_numpy.array2root(branch_Perp, inputD, treename='t' , mode='update')
    print("length branch_Long", len(branch_Long))
    print("length branch_Perp", len(branch_Perp))
    print('richtig, wenn auf a trainiert: -LongZ-pTZ', -NN_LongZ-mZ_r)
    print('richtig, wenn auf Z trainiert: LongZ-pTZ', np.add(NN_LongZ,-mZ_r))
    '''
    rfile = ROOT.TFile.Open(inputD)
    tree = rfile.Get('t')
    print([b.GetName() for b in tree.GetListOfBranches()])  # prints ['n_int', 'f_float', 'd_double']

    # remove the branch named 'd_double'
    #tree.SetBranchStatus('NN_LongZ', 0)
    #tree.SetBranchStatus('NN_PerpZ', 0)
    root_numpy.array2tree(branch_Long_Perp, tree=tree)

    # copy the tree into a new file
    rfile_out = ROOT.TFile.Open(outputDir+'Summer17_NN_Output.root', 'recreate')
    newtree = tree.CloneTree()
    newtree.Write()
    print([b.GetName() for b in newtree.GetListOfBranches()])
    '''
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
    print('Summe a_x enspricht prediction 0', np.sum(-a_x))
    print('Summe a_y enspricht prediction 0', np.sum(-a_y))
    print('Summe prediction 0', np.sum(NN_Output['MET_Predictions'][:,0]))
    print('Summe prediction 1', np.sum(NN_Output['MET_Predictions'][:,1]))



if __name__ == "__main__":
    inputDir = sys.argv[1]
    outputDir = sys.argv[2]
    NN_mode = sys.argv[3]
    plotsD = sys.argv[4]
    print(outputDir)
    NN_MVA = h5py.File("%s/NN_MVA_%s.h5"%(outputDir,NN_mode), "w")
    prepareOutput(outputDir, inputDir, NN_mode, plotsD)
