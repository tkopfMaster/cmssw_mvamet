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
    model.compile(
                    loss=loss_,
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

    # Set up preprocessing
    from sklearn.preprocessing import StandardScaler
    preprocessing_input = StandardScaler()
    preprocessing_input.fit(Inputs_train_train)
    preprocessing_target = StandardScaler()
    #Targets_train_val = Targets_train_val
    #Targets_train_train = Targets_train_train
    preprocessing_target.fit(Targets_train_train)

    # Train
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    history = model.fit(
        Inputs_train_train,
        Targets_train_train,
        #preprocessing_input.transform(Inputs_train_train),
        #preprocessing_target.transform(Targets_train_train),
        shuffle=True,
        batch_size=Inputs.shape[0],  #number of Inputs for loss calc
        epochs=1500,
        validation_data=(Inputs_train_val, Targets_train_val),
                         #preprocessing_input.transform(Inputs_train_val),
                         #preprocessing_target.transform(Targets_train_val)),
        callbacks=[
            ModelCheckpoint(save_best_only=True, filepath="%sMET_model_%s_%s_%s.h5"%(outputD,NN_mode, optim, loss_fct), verbose=1),
            EarlyStopping(patience=10000),

        ])
    model.save_weights("%sMET_weights_%s.h5"%(outputD,NN_mode))
    #predictions = preprocessing_target.inverse_transform(
    '''
    model.predict(
                preprocessing_input.transform(
                Inputs[:]
                )))

    print("Predictions 1", predictions[0:10,0])
    print("Target 1", Targets[0:10,0])
    print("Predictions 2",predictions[0:10,1])
    print("Target 2", Targets[0:10,1])
    print("Predictions 3",predictions[0:10,2])
    print("Target 3", Targets[0:10,2])
    plt.clf()
    plt.figure()
    plt.suptitle('x-Korrektur Prediction-Target ')
    plt.xlabel("${{p_T}^Z}_x$")
    plt.ylabel("Prediction-Target")
    heatmap, xedges, yedges = np.histogram2d(  map(np.subtract, predictions[:,0],  Targets[:,0]) , Targets[:,0], bins=50, range=[[np.percentile(Targets[:,0],5),np.percentile(Targets[:,0],95)],[np.percentile(map(np.subtract, predictions[:,0],  Targets[:,0]),5), np.percentile(map(np.subtract, predictions[:,0],  Targets[:,0]),95)]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    #plt.colorbar(heatmap.T)
    plt.legend()
    plt.savefig("/storage/b/tkopf/mvamet/plots/NN/HM_Pred-Tar_Tar_x.png")

    plt.clf()
    plt.figure()
    plt.suptitle('y-Korrektur: Prediction-Target ')
    plt.xlabel("$p_T^Z_y$")
    plt.ylabel("Prediction-Target")
    heatmap, xedges, yedges = np.histogram2d(  map(np.subtract, predictions[:,1],  Targets[:,1]) , Targets[:,1], bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    #plt.colorbar(heatmap.T)
    plt.legend()
    plt.savefig("/storage/b/tkopf/mvamet/plots/NN/HM_Pred-Tar_Tar_y.png")


    if NN_mode =='xyr' or NN_mode =='xyd' or NN_mode =='nr' or NN_mode =='xyd':
        plt.clf()
        plt.figure()
        plt.suptitle('y-Korrektur: Prediction-Target ')
        plt.xlabel("${p_T^Z}_y$")
        plt.ylabel("Prediction-Target")
        heatmap, xedges, yedges = np.histogram2d(  map(np.subtract, predictions[:,2],  Targets[:,2]) , Targets[:,2], bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        #plt.clf()
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        #plt.colorbar(heatmap.T)
        plt.legend()
        plt.savefig("/storage/b/tkopf/mvamet/plots/NN/HM_Pred-Tar_Tar_r.png")

    NN_Output = h5py.File("%sNN_Output_%s.h5"%(outputD,NN_mode), "w")
    dset = NN_Output.create_dataset("MET_Predictions", dtype='f', data=predictions)
    dset2 = NN_Output.create_dataset("MET_GroundTruth", dtype='f', data=Targets)
    dset3 = NN_Output.create_dataset(
        "MET_Loss", dtype='f', data=history.history['loss'])
    dset4 = NN_Output.create_dataset(
        "MET_Val_Loss", dtype='f', data=history.history['val_loss'])
    NN_Output.close()

    '''
    epochs = range(1, 1+len(history.history['loss']))
    plt.figure()
    plt.xlabel("Epochs all")
    plt.ylabel("Loss")
    plt.ylabel("Val_loss")
    plt.plot(epochs, history.history['loss'], 'g^', label="Loss")
    plt.plot(epochs, history.history['val_loss'], 'g-', label="Validation Loss")
    plt.legend(["Loss","Validation Loss"], loc='upper left')
    plt.savefig("%s/Loss.png"%(plotsD))

    epochs = range(1, 200)
    plt.figure()
    plt.xlabel("Epochs all")
    plt.ylabel("Loss")
    plt.ylabel("Val_loss")
    plt.plot(epochs, history.history['loss'][0:199], 'g^', label="Loss")
    plt.plot(epochs, history.history['val_loss'][0:199], 'g-', label="Validation Loss")
    plt.legend(["Loss","Validation Loss"], loc='upper left')
    plt.savefig("%s/Loss_CR.png"%(plotsD))

if __name__ == "__main__":
    outputDir = sys.argv[1]
    optim = str(sys.argv[2])
    loss_fct = str(sys.argv[3])
    NN_mode = sys.argv[4]
    plotsD = sys.argv[5]
    print(outputDir)
    writeInputs = h5py.File("%s/NN_model_%s_%s_%s.h5"%(outputDir,NN_mode, optim, loss_fct), "w")
    getModel(outputDir, optim, loss_fct, NN_mode, plotsD)
    writeInputs.close()
