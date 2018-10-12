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
from gaussian_1Training_wReweight import NNmodel
from numpy import s_
from sklearn.preprocessing import StandardScaler


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

'''
def NNmodel(x, reuse):
    ndim = 128
    with tf.variable_scope("model") as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable('w1', shape=(19,ndim), dtype=tf.float32,
                initializer=tf.glorot_normal_initializer())
        b1 = tf.get_variable('b1', shape=(ndim), dtype=tf.float32,
                initializer=tf.constant_initializer(0.0))
        w2 = tf.get_variable('w2', shape=(ndim, ndim), dtype=tf.float32,
                initializer=tf.glorot_normal_initializer())
        b2 = tf.get_variable('b2', shape=(ndim), dtype=tf.float32,
                initializer=tf.constant_initializer(0.0))

        w3 = tf.get_variable('w3', shape=(ndim, ndim), dtype=tf.float32,
                initializer=tf.glorot_normal_initializer())
        b3 = tf.get_variable('b3', shape=(ndim), dtype=tf.float32,
                initializer=tf.constant_initializer(0.0))
        w4 = tf.get_variable('w4', shape=(ndim, ndim), dtype=tf.float32,
                initializer=tf.glorot_normal_initializer())
        b4 = tf.get_variable('b4', shape=(ndim), dtype=tf.float32,
                initializer=tf.constant_initializer(0.0))

        w5 = tf.get_variable('w5', shape=(ndim, 2), dtype=tf.float32,
                initializer=tf.glorot_normal_initializer())
        b5 = tf.get_variable('b5', shape=(2), dtype=tf.float32,
                initializer=tf.constant_initializer(0.0))


    l1 = tf.nn.sigmoid(tf.add(b1, tf.matmul(x, w1)))
    l2 = tf.nn.sigmoid(tf.add(b2, tf.matmul(l1, w2)))
    l3 = tf.nn.sigmoid(tf.add(b3, tf.matmul(l2, w3)))
    l4 = tf.nn.sigmoid(tf.add(b4, tf.matmul(l3, w4)))
    logits = tf.add(b5, tf.matmul(l4, w5), name='output')
    return logits, logits
'''

def loadInputsTargets(outputD):
    InputsTargets = h5py.File("%sNN_Input_apply_%s.h5" % (outputD,NN_mode), "r")
    Input = np.row_stack((
                InputsTargets['PF'],
                InputsTargets['Track'],
                InputsTargets['NoPU'],
                InputsTargets['PUCorrected'],
                InputsTargets['PU'],
                InputsTargets['Puppi'],
                InputsTargets['NVertex']
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
    Inputs, Targets = loadInputsTargets(outputD)

    #get prediction
    x = tf.placeholder(tf.float32)
    logits, f = NNmodel(x, reuse=False)

    checkpoint_path = tf.train.latest_checkpoint(outputD)
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    #Preprocessing
    preprocessing_input = StandardScaler()
    #preprocessing_output = StandardScaler()
    preprocessing_input.fit(Inputs)
    #preprocessing_output.fit(Targets)
    firstthird = int(len(Inputs[:,0])/10)
    print("Laenge inputs", len(Inputs[:,0]))
    secondthird = int(len(Inputs[:,0])/3)*2
    predictions = []

    for i in range(0,10):
        start, end = i*firstthird, (i+1)*firstthird+1
        print("shape Inputs", Inputs.shape)
        predictions_i = sess.run(f, {x: preprocessing_input.transform(np.array_split(Inputs, 10, axis=0)[i])})
        print("shape np.array_split(Inputs, 10, axis=0)[i]", np.array_split(Inputs, 10, axis=0)[i].shape)
        if i==0:
            predictions = predictions_i
        else:
            predictions = np.append(predictions, predictions_i, axis=0)

    #print("TEST ", [Inputs[i:i + 10] for i in range(0, len(Inputs[:,0]), 10)])
    #predictions = [np.row_stack((predictions, sess.run(f, {x: preprocessing_input.transform(Inputs[i:i + 10])}))) for i in range(0, len(Inputs[:,0]), 10)]
    #prediction = preprocessing_output.inverse_transform(predictions)
    #get prediction CV
    x_CV = tf.placeholder(tf.float32)
    logits_CV, f_CV = NNmodel(x_CV, reuse=True)

    checkpoint_path_CV = tf.train.latest_checkpoint(outputD+"CV/")
    sess_CV = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.Saver()
    saver.restore(sess_CV, checkpoint_path_CV)
    predictions_CV = []
    for i in range(0,10):
        start, end = i*firstthird, (i+1)*firstthird+1
        print("shape Inputs", Inputs.shape)
        predictions_i = sess.run(f_CV, {x_CV: preprocessing_input.transform(np.array_split(Inputs, 10, axis=0)[i])})
        print("shape np.array_split(Inputs, 10, axis=0)[i]", np.array_split(Inputs, 10, axis=0)[i].shape)
        if i==0:
            predictions_CV = predictions_i
        else:
            predictions_CV = np.append(predictions_CV, predictions_i, axis=0)

    #merge both predictions
    Test_Idx2 = h5py.File("%sTest_Idx_CV_%s.h5" % (outputDir, NN_mode), "r")
    Test_Idx = Test_Idx2["Test_Idx"].value.astype(int)
    print("shape predictions", predictions.shape)
    print("shape predictions_CV", predictions_CV.shape)
    print(" max Test_Idx", max(Test_Idx))
    predictions[Test_Idx,:] = predictions_CV[Test_Idx,:]


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
    print("It has been {0} seconds since application of the NN model started".format(now - program_starts))

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
