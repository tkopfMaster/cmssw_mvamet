#!/usr/bin/env python

import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np

def loadInputs(outputD):
    InputsTargets = h5py.File("%sNN_Input.h5" % (outputD), "r")


def getModel():
    inputs = np.column_stack(
        (dset_PF[:, 0:3], dset_Track[:, 0:3], dset_NoPU[:, 0:3],
         dset_PUcorr[:, 0:3], dset_PU[:, 0:3]))


    targets = dset_PF[:, 4]


    print("Loaded MET dataset with {} entries.".format(inputs.shape))
    print("Example entry: {}".format(inputs[0]))
    print("Loaded MET targets dataset with {} entries.".format(targets.shape))
    print("Example targets entry: {}".format(targets[0:10]))

    # Define model
    model = Sequential()

    model.add(
        Dense(
            1000,  # Number of nodes
            kernel_initializer="glorot_normal",  # Initialization
            activation="tanh",  # Activation
            input_dim=inputs.shape[1]))  # Shape of inputs (only needed for the first layer)

    model.add(
        Dense(
            1,  #Dimension des Output Space --> 1 fuer Regression
            kernel_initializer="glorot_uniform",
            activation="linear"))  # Regressions

    model.summary()

    # Set loss, optimizer and evaluation metrics
    model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mean_absolute_error", "mean_squared_error"])

    # Split dataset in training and testing
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, targets, test_size=0.25, random_state=1234)



if __name__ == "__main__":
    outputDir = sys.argv[1]
    print(outputDir)
    writeInputs = h5py.File("%s/NN_Model.h5"%outputDir, "w")
	getModel()
