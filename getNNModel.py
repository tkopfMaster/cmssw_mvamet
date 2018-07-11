#!/usr/bin/env python

import h5py
import matplotlib as mpl
mpl.use('Agg')
#import matplotlib.pyplot as plt
import sys
import numpy as np
from os import environ

pTCut = 30
#NN_mode = 'xyr'


def loadInputsTargets(outputD):
    InputsTargets = h5py.File("%sNN_Input_training_%s.h5" % (outputD,NN_mode), "r")
    norm = np.sqrt(np.multiply(InputsTargets['Target'][:,0],InputsTargets['Target'][:,0]) + np.multiply(InputsTargets['Target'][:,1],InputsTargets['Target'][:,1]))

    pTCut_Idx = norm.index[norm > pTCut].tolist()
    Target =  InputsTargets['Target']
    Input = np.row_stack((
                InputsTargets['PF'][pTCut_Idx,:],
                InputsTargets['Track'][pTCut_Idx,:],
                InputsTargets['NoPU'][pTCut_Idx,:],
                InputsTargets['PUCorrected'][pTCut_Idx,:],
                InputsTargets['PU'][pTCut_Idx,:],
                InputsTargets['Puppi'][pTCut_Idx,:]
                ))


    return (np.transpose(Input), np.transpose(Target))


def getModel(outputD, optimiz, loss_, NN_mode, plotsD):
    Inputs, Targets = loadInputsTargets(outputD)

    print("Loaded MET dataset with {} entries.".format(Inputs.shape))
    print("Example entry: {}".format(Inputs[0]))
    print("Loaded MET Targets dataset with {} entries.".format(Targets.shape))
    print("Example Targets entry: {}".format(Targets[0:10,:]))

    print('Zeile 38')
    # Select TensorFlow as backend for Keras
    environ["KERAS_BACKEND"] = "tensorflow"
    np.random.seed(1234)  #immer vor keras
    from keras.utils import np_utils
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout
    from keras.optimizers import Adam
    print('Zeile 46')
    # Define model
    model = Sequential()
    model.add(
        Dense(
            100,  # Number of nodes
            kernel_initializer="glorot_normal",  # Initialization
            activation="relu",  # Activation
            input_dim=Inputs.shape[1]))  # Shape of Inputs (only needed for the first layer)
    if NN_mode == 'xyr' or NN_mode == 'nr' or NN_mode == 'xyra' or NN_mode == 'xyd':
        model.add(
                Dense(
                    3,  #Dimension des Output Space --> 1 fuer Regression
                    kernel_initializer="glorot_uniform",
                    activation="linear"))  # Regressions
        model.summary()
    else:
        model.add(
                Dense(
                    2,  #Dimension des Output Space --> 1 fuer Regression
                    kernel_initializer="glorot_uniform",
                    activation="linear"))  # Regressions
        model.summary()
    print('Zeile 69')
    # Set loss, optimizer and evaluation metrics
    from own_loss_functions import mean_squared_error_r, perp_long_error
    if "mean_squared_error_r" in loss_: loss_function = mean_squared_error_r
    elif "perp_long_error" in loss_: loss_function = perp_long_error
    else: loss_function = loss_
    print('Zeile 75')
    model.compile(
                    loss=loss_function,
                    optimizer=optimiz)
                    #metrics=["mean_absolute_error", "mean_squared_error"])

    # Split dataset in training and testing
    print('Zeile 82')
    from sklearn.model_selection import train_test_split
    print('Zeile 84')
    Inputs_train, Inputs_test, Targets_train, Targets_test = train_test_split(
        Inputs, Targets,
        test_size=0.90,
        random_state=1234)
    Inputs_train_train, Inputs_train_val, Targets_train_train, Targets_train_val = train_test_split(
        Inputs_train, Targets_train,
        test_size=0.5,
        random_state=1234)

    print('Zeile 91')
    # Train
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    history = model.fit(
        Inputs_train_train,
        Targets_train_train,
        shuffle=True,
        batch_size=int(Inputs_train.shape[0]/100),  #number of Inputs for loss calc
        epochs=100,
        validation_data=(Inputs_train_val, Targets_train_val),
        callbacks=[
            ModelCheckpoint(save_best_only=True, filepath="%sMET_model_%s_%s_%s.h5"%(outputD,NN_mode, optim, loss_fct), verbose=1),
            EarlyStopping(patience=10000),

        ])


    dset = NN_Output.create_dataset("loss", dtype='f', data=history.history['loss'])
    dset2 = NN_Output.create_dataset("val_loss", dtype='f', data=history.history['val_loss'])
    NN_Output.close()

if __name__ == "__main__":
    outputDir = sys.argv[1]
    optim = str(sys.argv[2])
    loss_fct = str(sys.argv[3])
    NN_mode = sys.argv[4]
    plotsD = sys.argv[5]
    print(outputDir)
    NN_Output = h5py.File("%sNN_Output_%s.h5"%(outputDir,NN_mode), "w")
    getModel(outputDir, optim, loss_fct, NN_mode, plotsD)
