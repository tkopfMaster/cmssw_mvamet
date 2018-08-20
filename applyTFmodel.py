#!/usr/bin/env python
import time
import h5py
import matplotlib as mpl
mpl.use('Agg')
#import matplotlib.pyplot as plt
import sys
import numpy as np
from os import environ
import root_numpy as rnp
import tensorflow as tf
from gaussian import NNmodel

#NN_mode='xyr'

def loadInputsTargets(outputD):
    InputsTargets = h5py.File("%sNN_Input_apply_%s.h5" % (outputD,NN_mode), "r")
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


def loadInputs(inputD):
    InputsTargets = h5py.File("%sNN_Input_apply_%s.h5" % (inputD, NN_mode), "r")
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
    InputsTargets = h5py.File("%sNN_Input_apply_%s.h5" % (inputD, NN_mode), "r")

    Target =  InputsTargets['Target']
    return (np.transpose(Target))

def loadBosonPt(inputD):
    InputsTargets = h5py.File("%sNN_Input_apply_%s.h5" % (inputD, NN_mode), "r")
    Target =  np.squeeze(InputsTargets['Boson_Pt'][:])
    return (np.transpose(Target))



def applyModel(outputD, inputD, NN_mode, optimiz, loss_):
    program_starts = time.time()
    from keras.models import load_model
    from own_loss_functions import mean_squared_error_r, perp_long_error
    if "mean_squared_error_r" in loss_:
        import keras.losses
        keras.losses.mean_squared_error_r = mean_squared_error_r
        model = load_model("%sMET_model_%s_%s_%s.h5"%(outputD,NN_mode, optim, loss_))
    elif "perp_long_error" in loss_:
        import keras.losses
        keras.losses.mean_squared_error_r = perp_long_error
        model = load_model("%sMET_model_%s_%s_%s.h5"%(outputD,NN_mode, optim, loss_))
    else:
        loss_function = loss_
        #sess = tf.Session()
        #saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(outputD)+".meta")
        #saver.restore(sess, tf.train.latest_checkpoint(outputD))
        #node = tf.get_default_graph().get_tensor_by_name("logits:0")
    Inputs, Targets = loadInputsTargets(outputD)

    x = tf.placeholder(tf.float32)
    logits, f = NNmodel(x, reuse=False)

    checkpoint_path = tf.train.latest_checkpoint(outputD)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    predictions = sess.run(f, {x: Inputs})

    '''
    for i in range(len(Inputs[:,0])//1000000):
        x = tf.placeholder(tf.float32)
        checkpoint_path = tf.train.latest_checkpoint(outputD)
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)
        logits, f = NNmodel(x, reuse=False)
        predictions.append(sess.run(f, {x: Inputs[i:((i+1)*1000000),:]}))
    rest = len(Inputs[:,0])%i
    x = tf.placeholder(tf.float32)
    logits, f = NNmodel(x, reuse=False)
    print('i*1000000+rest', i*1000000+rest)
    print('len(Inputs[:,0])', len(Inputs[:,0]))
    predictions.append(sess.run(f, {x: Inputs[i:(i*1000000+rest),:]}))
    '''

    #predictions = model.predict(Inputs[:])
    print('len(Targets)', len(Targets))
    print('len(predictions)', len(predictions))
    print("predictions in apply NN ", predictions	)
    print('Mean x deviation', np.mean(np.subtract(predictions[:,0], Targets[:,0])))
    print('Std x deviation', np.std(np.subtract(predictions[:,0], Targets[:,0])))
    print('Mean y deviation', np.mean(np.subtract(predictions[:,1], Targets[:,1])))
    print('Std y deviation', np.std(np.subtract(predictions[:,1], Targets[:,1])))
    dset = NN_Output_applied.create_dataset("MET_Predictions", dtype='f', data=predictions)
    dset2 = NN_Output_applied.create_dataset("MET_GroundTruth", dtype='f', data=Targets)
    dset3 = NN_Output_applied.create_dataset('Boson_Pt', dtype='f', data=loadBosonPt(outputD)[:])
    NN_Output_applied.close()
    now = time.time()
    print("It has been {0} seconds since the program started".format(now - program_starts))

if __name__ == "__main__":
    inputDir = sys.argv[1]
    outputDir = sys.argv[2]
    optim = str(sys.argv[3])
    loss_fct = str(sys.argv[4])
    NN_mode = sys.argv[5]
    print(outputDir)
    #writeInputs = h5py.File("%sNN_Input_%s.h5"%(outputDir,NN_mode), "r+")
    NN_Output_applied = h5py.File("%sNN_Output_applied_%s.h5"%(outputDir,NN_mode), "w")
    applyModel(outputDir, inputDir, NN_mode,  optim, loss_fct)