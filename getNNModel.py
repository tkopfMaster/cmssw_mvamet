#!/usr/bin/env python

import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
from os import environ


#NN_mode = 'xyr'


def loadInputsTargets(outputD):
    InputsTargets = h5py.File("%sNN_Input_%s.h5" % (outputD,NN_mode), "r")
    Input = np.row_stack((
                InputsTargets['PF'],
                InputsTargets['Track'],
                InputsTargets['NoPU'],
                InputsTargets['PUCorrected'],
                InputsTargets['PU'],
                InputsTargets['Puppi']
                ))

    Target =  InputsTargets['Target']
    return (np.transpose(Input), np.transpose(Target))


def getModel(outputD, optimiz, loss_, NN_mode, plotsD):
    Inputs, Targets = loadInputsTargets(outputD)

    print("Loaded MET dataset with {} entries.".format(Inputs.shape))
    print("Example entry: {}".format(Inputs[0]))
    print("Loaded MET Targets dataset with {} entries.".format(Targets.shape))
    print("Example Targets entry: {}".format(Targets[0:10,:]))


    # Select TensorFlow as backend for Keras
    environ["KERAS_BACKEND"] = "tensorflow"
    np.random.seed(1234)  #immer vor keras
    from keras.utils import np_utils
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout
    from keras.optimizers import Adam

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

    # Set loss, optimizer and evaluation metrics
    from own_loss_functions import mean_squared_error_r
    if "mean_squared_error_r" in loss_: loss_function = mean_squared_error_r
    else: loss_function = loss_
    model.compile(
                    loss=loss_function,
                    optimizer=optimiz)
                    #metrics=["mean_absolute_error", "mean_squared_error"])

    # Split dataset in training and testing
    from sklearn.model_selection import train_test_split
    Inputs_train, Inputs_test, Targets_train, Targets_test = train_test_split(
        Inputs, Targets,
        test_size=0.90,
        random_state=1234)
    Inputs_train_train, Inputs_train_val, Targets_train_train, Targets_train_val = train_test_split(
        Inputs_train, Targets_train,
        test_size=0.5,
        random_state=1234)


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



if __name__ == "__main__":
    outputDir = sys.argv[1]
    optim = str(sys.argv[2])
    loss_fct = str(sys.argv[3])
    NN_mode = sys.argv[4]
    plotsD = sys.argv[5]
    print(outputDir)
    getModel(outputDir, optim, loss_fct, NN_mode, plotsD)
