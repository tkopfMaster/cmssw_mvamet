#!/usr/bin/env python

import h5py
import matplotlib as mpl
mpl.use('Agg')
#import matplotlib.pyplot as plt
import sys
import numpy as np
from os import environ
import root_numpy as rnp

#NN_mode='xyr'


def loadInputs(inputD):
    InputsTargets = h5py.File("%sNN_Input_%s.h5" % (inputD, NN_mode), "r")
    Input = np.row_stack((
                InputsTargets['PF'],
                InputsTargets['Track'],
                InputsTargets['NoPU'],
                InputsTargets['PUCorrected'],
                InputsTargets['PU'],
                InputsTargets['Puppi']
                ))

    return (np.transpose(Input))

def loadTargets(inputD):
    InputsTargets = h5py.File("%sNN_Input_%s.h5" % (inputD, NN_mode), "r")

    Target =  InputsTargets['Target']
    return (np.transpose(Target))

def applyModel(outputD, inputD, NN_mode, optimiz, loss_):
    print('Zeile34')
    from keras.models import load_model
    print('zeile35')
    model = load_model("%sMET_model_%s_%s_%s.h5"%(outputD,NN_mode, optim, loss_fct))
    Inputs=loadInputs(outputD)
    Targets=loadTargets(outputD)
    # Set up preprocessing
    from keras.utils import np_utils
    '''
    print('Zeile 40')
    from sklearn.preprocessing import StandardScaler
    preprocessing_input = StandardScaler()
    preprocessing_input.fit(Inputs)
    preprocessing_target = StandardScaler()
    preprocessing_target.fit(Targets)

    predictions = preprocessing_target.inverse_transform(
    model.predict(
                preprocessing_input.transform(
                Inputs
    )))
    '''
    predictions = model.predict(Inputs)
    print("predictions in apply NN ", predictions	)
    dset = NN_Output.create_dataset("MET_Predictions", dtype='f', data=predictions)
    dset2 = NN_Output.create_dataset("MET_GroundTruth", dtype='f', data=Targets)
    #dset3 = NN_Output.create_dataset(
    #    "MET_Loss", dtype='f', data=history.history['loss'])
    #dset4 = NN_Output.create_dataset(
    #    "MET_Val_Loss", dtype='f', data=history.history['val_loss'])
    NN_Output.close()

if __name__ == "__main__":
    inputDir = sys.argv[1]
    outputDir = sys.argv[2]
    optim = str(sys.argv[3])
    loss_fct = str(sys.argv[4])
    NN_mode = sys.argv[5]
    print(outputDir)
    NN_Output = h5py.File("%sNN_Output_%s.h5"%(outputDir,NN_mode), "w")
    applyModel(outputDir, inputDir, NN_mode,  optim, loss_fct)
