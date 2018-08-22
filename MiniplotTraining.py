import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import pandas as pd
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from getPlotsOutputclean import loadData, loadData_woutGBRT
from getResponse import getResponse, getResponseIdx
from prepareInput import pol2kar_x, pol2kar_y, kar2pol, pol2kar, angularrange


pTMin, pTMax = 20,200

def plotTraining(outputD, optim, loss_fct, NN_mode, plotsD, rootOutput, PhysicsProcess, Target_Pt, Target_Phi, Test_Idx):
    IndTest = sorted([int(x) for x in Test_Idx])



    print("Jetzt kommt InputsTargets_hdf5")
    if NN_mode == 'xy':
        #Load Inputs and Targets with Name

        InputsTargets_hdf5 = h5py.File("%sNN_Input_apply_%s.h5" % (outputD,NN_mode), "r")
        print("Keys: %s" % InputsTargets_hdf5.keys())
        keys = InputsTargets_hdf5.keys()
        values = [InputsTargets_hdf5[k].value for k in keys]
        #InputsTargets2 = pd.DataFrame(index=np.arange(NN_Output_applied["MET_GroundTruth"].shape))
        print("length index", len(np.arange(len(InputsTargets_hdf5['Target'][0,:]))))
        print("length values", len(InputsTargets_hdf5['Target'][0,:]))
        InputsTargets = pd.Series()
        Norm_pT = np.sqrt(np.multiply(InputsTargets_hdf5['Target'][0,:], InputsTargets_hdf5['Target'][0,:]) + np.multiply(InputsTargets_hdf5['Target'][1,:], InputsTargets_hdf5['Target'][1,:]))
        print(np.sort(np.array(Test_Idx, dtype=np.int64), axis=None))
        for k, v in zip(keys, values):
                InputsTargets[k] = v
                print(type(InputsTargets[k]))
                print(k)
        print(type(InputsTargets))
        #boolInd = np.zeros(len(InputsTargets_hdf5['Target'][0,IndTest]))
        #boolInd[int(np.asarray(Test_Idx))] = 1
        #InputsTargets = [InputsTargets[k].loc[np.sort(np.array(Test_Idx, dtype=np.int64), axis=None)] for k in InputsTargets.index]
        print("InputsTargets geschafft, Laenge", len(InputsTargets['Target'][0,IndTest]))
        #InputsTargets = InputsTargets_hdf5[Test_Idx]


        #print('InputsTargets', InputsTargets.shape())


if __name__ == "__main__":
    outputDir = sys.argv[1]
    optim = str(sys.argv[2])
    loss_fct = str(sys.argv[3])
    NN_mode = sys.argv[4]
    plotsD = sys.argv[5]
    PhysicsProcess = sys.argv[6]
    rootInput = sys.argv[7]

    Test_Idx2 = h5py.File("%sTest_Idx_%s.h5" % (outputDir, NN_mode), "r")
    Test_Idx = Test_Idx2["Test_Idx"]
    IndTest = sorted([int(x) for x in Test_Idx])



    print("Jetzt kommt InputsTargets_hdf5")

    InputsTargets_hdf5 = h5py.File("%sNN_Input_apply_%s.h5" % (outputDir,NN_mode), "r")
    print("Keys: %s" % InputsTargets_hdf5.keys())
    keys = InputsTargets_hdf5.keys()
    values = [InputsTargets_hdf5[k].value for k in keys]
    #InputsTargets2 = pd.DataFrame(index=np.arange(NN_Output_applied["MET_GroundTruth"].shape))
    print("length index", len(np.arange(len(InputsTargets_hdf5['Target'][0,:]))))
    print("length values", len(InputsTargets_hdf5['Target'][0,:]))
    InputsTargets = pd.Series()
    Norm_pT = np.sqrt(np.multiply(InputsTargets_hdf5['Target'][0,:], InputsTargets_hdf5['Target'][0,:]) + np.multiply(InputsTargets_hdf5['Target'][1,:], InputsTargets_hdf5['Target'][1,:]))
    print(np.sort(np.array(Test_Idx, dtype=np.int64), axis=None))
    for k, v in zip(keys, values):
            InputsTargets[k] = v
            print(type(InputsTargets[k]))
            print(k)
    print(type(InputsTargets))
    #boolInd = np.zeros(len(InputsTargets_hdf5['Target'][0,IndTest]))
    #boolInd[int(np.asarray(Test_Idx))] = 1
    #InputsTargets = [InputsTargets[k].loc[np.sort(np.array(Test_Idx, dtype=np.int64), axis=None)] for k in InputsTargets.index]
    print("InputsTargets geschafft, Laenge", len(InputsTargets['Target'][0,IndTest]))
