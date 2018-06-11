#!/usr/bin/env python

import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
from os import environ
import root_numpy
from rootpy.tree import Tree, TreeModel, FloatCol, IntCol
from rootpy.io import root_open
import ROOT

NN_mode='xyr'

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

def loadData(fName, NN_mode):
    treeName = 't'
    arrayName = rnp.root2array(fName, branches=['Boson_Pt',
        'recoilslimmedMETs_Pt'],)
    DFName = pd.DataFrame.from_records(arrayName.view(np.recarray))
    return(DFName)

def prepareOutput(outputD, inputD, NN_mode):
    NN_Output = h5py.File("%sNN_Output_%s.h5"%(outputD,NN_mode), "r+")
    if NN_mode == 'xyr' or NN_mode == 'nr':
        mZ_x, mZ_y, mZ_r = NN_Output['MET_GroundTruth'][:,0], NN_Output['MET_GroundTruth'][:,1], NN_Output['MET_GroundTruth'][:,2]
        a_x, a_y, a_r = -NN_Output['MET_Predictions'][:,0], -NN_Output['MET_Predictions'][:,1], NN_Output['MET_Predictions'][:,2]
    elif NN_mode == 'xyd':
        PF_Z_pT = loadData(inputD)
        mZ_x, mZ_y, mZ_r = NN_Output['MET_GroundTruth'][:,0], NN_Output['MET_GroundTruth'][:,1], PF_Z_pT['Boson_Pt']
        a_x, a_y, a_r = -NN_Output['MET_Predictions'][:,0], -NN_Output['MET_Predictions'][:,1], NN_Output['MET_Predictions'][:,2]+PF_Z_pT['recoilslimmedMETs_Pt']
    else:
        mZ_x, mZ_y = NN_Output['MET_GroundTruth'][:,0], NN_Output['MET_GroundTruth'][:,1]
        a_x, a_y = -NN_Output['MET_Predictions'][:,0], -NN_Output['MET_Predictions'][:,1]
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

    print("a_r,a_phi", a_r,a_phi)

    NN_LongZ, NN_PerpZ= pol2kar(a_r,(a_phi+np.pi)-mZ_phi)
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
    print('richtig, wenn auf Z trainiert: LongZ-pTZ', NN_LongZ-mZ_r)
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






if __name__ == "__main__":
    inputDir = sys.argv[1]
    outputDir = sys.argv[2]
    NN_mode = sys.argv[3]
    print(outputDir)
    NN_MVA = h5py.File("%s/NN_MVA_%s.h5"%(outputDir,NN_mode), "w")
    prepareOutput(outputDir, inputDir, NN_mode)
