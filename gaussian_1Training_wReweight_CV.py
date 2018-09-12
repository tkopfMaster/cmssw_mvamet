'''debug gpu usage
import tensorflow as tf
print("bla blubb", tf.test.is_built_with_cuda())

import tensorflow
from tensorflow.python.client import device_lib
print("Auf der deepthought", device_lib.list_local_devices())

print(tensorflow.__version__)
sess = tensorflow.Session()
print(" info gpu ", tensorflow.Session(config=tensorflow.ConfigProto(log_device_placement=True)))
'''

import numpy as np
np.random.seed(1234)
from sklearn.metrics import roc_curve, auc
from gaussian_1Training_wReweight import loadInputsTargetsWeights
import tensorflow as tf
import datetime
import sys
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D as plt3d
import time
import sys

reweighting = True

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

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


    l1 = tf.nn.relu(tf.add(b1, tf.matmul(x, w1)))
    l2 = tf.nn.relu(tf.add(b2, tf.matmul(l1, w2)))
    l3 = tf.nn.relu(tf.add(b3, tf.matmul(l2, w3)))
    l4 = tf.nn.relu(tf.add(b4, tf.matmul(l3, w4)))
    logits = tf.add(b5, tf.matmul(l4, w5), name='output')
    return logits, logits

def moving_average(data_set, periods):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')

def Names(outputD, NN_mode):
    InputsTargets = h5py.File("%sNN_Input_training_%s.h5" % (outputD,NN_mode), "r")
    return (InputsTargets.keys)


def costResolution_perp(y_true,y_pred, weight):
    a_=tf.sqrt(tf.square(y_pred[:,0])+tf.square(y_pred[:,1]))
    pZ = tf.sqrt(tf.square(y_true[:,0])+tf.square(y_true[:,1]))
    alpha_a=tf.atan2(y_pred[:,1],y_pred[:,0])
    alpha_Z=tf.atan2(y_true[:,1],y_true[:,0])
    alpha_diff=tf.subtract(alpha_a,alpha_Z)
    u_perp = tf.sin(alpha_diff)*a_
    u_long = tf.cos(alpha_diff)*a_
    u_perp_ = tf.square(u_perp)+tf.square(u_long-pZ)

    cost= tf.multiply(u_perp_,weight)
    return tf.reduce_mean(cost)


def costResolution_para(y_true,y_pred, weight):
    a_=tf.sqrt(tf.square(y_pred[:,0])+tf.square(y_pred[:,1]))
    pZ = tf.sqrt(tf.square(y_true[:,0])+tf.square(y_true[:,1]))
    alpha_a=tf.atan2(y_pred[:,1],y_pred[:,0])
    alpha_Z=tf.atan2(y_true[:,1],y_true[:,0])
    alpha_diff=tf.subtract(alpha_a,alpha_Z)
    u_perp = tf.sin(alpha_diff)*a_
    u_long = tf.cos(alpha_diff)*a_
    Response = tf.divide(u_long,pZ)
    Resolution_para = u_long-pZ

    cost = tf.multiply(tf.square(Resolution_para),weight)
    return tf.reduce_mean(cost)

def costExpected(y_true,y_pred, weight):
    a_=tf.sqrt(tf.square(y_pred[:,0])+tf.square(y_pred[:,1]))
    pZ = tf.sqrt(tf.square(y_true[:,0])+tf.square(y_true[:,1]))
    alpha_a=tf.atan2(y_pred[:,1],y_pred[:,0])
    alpha_Z=tf.atan2(y_true[:,1],y_true[:,0])
    alpha_diff=tf.subtract(alpha_a,alpha_Z)
    u_perp = tf.sin(alpha_diff)*a_
    u_perp_ = tf.sin(alpha_diff)*pZ
    u_long = tf.cos(alpha_diff)*a_
    Response = tf.divide(u_long,pZ)
    Resolution_para = u_long-pZ

    cost = tf.multiply(tf.square(Resolution_para)+tf.square(u_perp_),weight)
    return tf.reduce_mean(cost)

def costExpectedPz(y_true,y_pred, weight):
    a_=tf.sqrt(tf.square(y_pred[:,0])+tf.square(y_pred[:,1]))
    pZ = tf.sqrt(tf.square(y_true[:,0])+tf.square(y_true[:,1]))
    alpha_a=tf.atan2(y_pred[:,1],y_pred[:,0])
    alpha_Z=tf.atan2(y_true[:,1],y_true[:,0])
    alpha_diff=tf.subtract(alpha_a,alpha_Z)
    u_perp = tf.sin(alpha_diff)*a_
    u_perp_ = tf.sin(alpha_diff)*pZ
    u_long = tf.cos(alpha_diff)*a_
    Response = tf.divide(u_long,pZ)
    Resolution_para = u_long-pZ

    cost = tf.multiply(tf.square(tf.multiply(Resolution_para,pZ))+tf.square(u_perp_),weight)
    return tf.reduce_mean(cost)

def cost10Expected(y_true,y_pred, weight):
    a_=tf.sqrt(tf.square(y_pred[:,0])+tf.square(y_pred[:,1]))
    pZ = tf.sqrt(tf.square(y_true[:,0])+tf.square(y_true[:,1]))
    alpha_a=tf.atan2(y_pred[:,1],y_pred[:,0])
    alpha_Z=tf.atan2(y_true[:,1],y_true[:,0])
    alpha_diff=tf.subtract(alpha_a,alpha_Z)
    u_perp = tf.sin(alpha_diff)*a_
    u_perp_ = tf.sin(alpha_diff)*pZ
    u_long = tf.cos(alpha_diff)*a_
    Response = tf.divide(u_long,pZ)
    Resolution_para = u_long-pZ

    cost = tf.multiply(40*tf.square(Resolution_para)+tf.square(u_perp_),weight)
    return tf.reduce_mean(cost)

def costExpectedRel(y_true,y_pred, weight):
    a_=tf.sqrt(tf.square(y_pred[:,0])+tf.square(y_pred[:,1]))
    pZ = tf.sqrt(tf.square(y_true[:,0])+tf.square(y_true[:,1]))
    alpha_a=tf.atan2(y_pred[:,1],y_pred[:,0])
    alpha_Z=tf.atan2(y_true[:,1],y_true[:,0])
    alpha_diff=tf.subtract(alpha_a,alpha_Z)
    u_perp = tf.sin(alpha_diff)*a_
    u_perp_ = tf.sin(alpha_diff)*pZ
    u_long = tf.cos(alpha_diff)*a_
    Response = tf.divide(u_long,pZ)
    Resolution_para = u_long-pZ

    cost = tf.square(Response-1)
    return tf.reduce_mean(cost)

def costExpectedRelAsy(y_true,y_pred, weight):
    a_=tf.sqrt(tf.square(y_pred[:,0])+tf.square(y_pred[:,1]))
    pZ = tf.sqrt(tf.square(y_true[:,0])+tf.square(y_true[:,1]))
    alpha_a=tf.atan2(y_pred[:,1],y_pred[:,0])
    alpha_Z=tf.atan2(y_true[:,1],y_true[:,0])
    alpha_diff=tf.subtract(alpha_a,alpha_Z)
    u_perp = tf.sin(alpha_diff)*a_
    u_perp_ = tf.sin(alpha_diff)*pZ
    u_long = tf.cos(alpha_diff)*a_
    Response = tf.divide(u_long,pZ)
    Resolution_para = u_long-pZ
    Response_over1 = tf.reduce_sum(tf.square(tf.nn.relu(Response-1)))
    Response_under1 = tf.reduce_sum(tf.square(tf.nn.relu(1-Response)))
    Response_Diff = tf.reduce_mean(tf.abs(Response_over1-Response_under1))
    cost = tf.square(Response_over1)+tf.square(Response_under1)
    return cost+Response_Diff


def costExpectedRelAsy2(y_true,y_pred, weight):
    a_=tf.sqrt(tf.square(y_pred[:,0])+tf.square(y_pred[:,1]))
    pZ = tf.sqrt(tf.square(y_true[:,0])+tf.square(y_true[:,1]))
    alpha_a=tf.atan2(y_pred[:,1],y_pred[:,0])
    alpha_Z=tf.atan2(y_true[:,1],y_true[:,0])
    alpha_diff=tf.subtract(alpha_a,alpha_Z)
    u_perp = tf.sin(alpha_diff)*a_
    u_perp_ = tf.sin(alpha_diff)*pZ
    u_long = tf.cos(alpha_diff)*a_
    Response = tf.divide(u_long,pZ)
    Resolution_para = u_long-pZ
    Response_over1 = tf.reduce_sum(tf.square(tf.nn.relu(Response-1)))
    Response_under1 = tf.reduce_sum(tf.square(tf.nn.relu(1-Response)))
    return Response_over1+Response_under1

def costExpectedRelAbs(y_true,y_pred, weight):
    a_=tf.sqrt(tf.square(y_pred[:,0])+tf.square(y_pred[:,1]))
    pZ = tf.sqrt(tf.square(y_true[:,0])+tf.square(y_true[:,1]))
    alpha_a=tf.atan2(y_pred[:,1],y_pred[:,0])
    alpha_Z=tf.atan2(y_true[:,1],y_true[:,0])
    alpha_diff=tf.subtract(alpha_a,alpha_Z)
    u_perp = tf.sin(alpha_diff)*a_
    u_perp_ = tf.sin(alpha_diff)*pZ
    u_long = tf.cos(alpha_diff)*a_
    Response = tf.divide(u_long,pZ)
    Resolution_para = u_long-pZ

    cost = tf.multiply(tf.square(tf.divide(Resolution_para,pZ))+tf.square(tf.sin(alpha_diff)),weight)
    return tf.reduce_mean(cost)

def costResolutions(y_true,y_pred, weight):
    a_=tf.sqrt(tf.square(y_pred[:,0])+tf.square(y_pred[:,1]))
    pZ = tf.sqrt(tf.square(y_true[:,0])+tf.square(y_true[:,1]))
    alpha_a=tf.atan2(y_pred[:,1],y_pred[:,0])
    alpha_Z=tf.atan2(y_true[:,1],y_true[:,0])
    alpha_diff=tf.subtract(alpha_a,alpha_Z)
    u_perp = tf.sin(alpha_diff)*a_
    u_perp_ = tf.sin(alpha_diff)*pZ
    u_long = tf.cos(alpha_diff)*a_
    Response = tf.divide(u_long,pZ)
    Resolution_para = u_long-pZ

    cost = tf.multiply(tf.square(Resolution_para)+tf.square(u_perp),weight)
    return tf.reduce_mean(cost)


def costResponse(y_true,y_pred, weight):
    a_=tf.sqrt(tf.square(y_pred[:,0])+tf.square(y_pred[:,1]))
    pZ = tf.sqrt(tf.square(y_true[:,0])+tf.square(y_true[:,1]))
    alpha_a=tf.atan2(y_pred[:,1],y_pred[:,0])
    alpha_Z=tf.atan2(y_true[:,1],y_true[:,0])
    alpha_diff=tf.subtract(alpha_a,alpha_Z)
    u_perp = tf.sin(alpha_diff)*a_
    u_long = tf.cos(alpha_diff)*a_
    Response = tf.divide(u_long,pZ)

    cost= tf.multiply(tf.square(tf.multiply(Response-1,pZ)), weight)
    return tf.reduce_mean(cost)



def costMSE(y_true,y_pred, weight):
    MSE_x=tf.square(y_pred[:,0]-y_true[:,0])
    MSE_y = tf.square(y_pred[:,1]-y_true[:,1])

    cost = tf.multiply(MSE_x+MSE_y, weight)
    return tf.reduce_mean(cost)




def getModel(outputDir, optim, loss_fct, NN_mode, plotsD):
    start = time.time()
    Inputs, Targets, Weights = loadInputsTargetsWeights(outputDir, NN_mode)

    Boson_Pt = np.sqrt(np.square(Targets[:,0])+np.square(Targets[:,1]))
    num_events = Inputs.shape[0]
    print('Number of events in get model ', num_events)
    Test_Idx2 = h5py.File("%sTest_Idx_%s.h5" % (outputDir, NN_mode), "r")
    training_idx = (Test_Idx2["Test_Idx"].value).astype(int)
    #train_test_splitter = 0.8
    #training_idx = np.random.choice(np.arange(Inputs.shape[0]), int(Inputs.shape[0]*train_test_splitter), replace=False)
    print('random training index length', training_idx.shape)
    print('inputs shape', Inputs.shape)
    print('First 10 Training Idxs', training_idx[0:10])

    #Write Test Idxs
    test_idx = np.setdiff1d(  np.arange(Inputs.shape[0]), training_idx)
    dset = Test_Idx.create_dataset("Test_Idx",  dtype='f', data=test_idx)
    Test_Idx.close()


    # Test if something's not right if the shapes don't match
    if not (len(test_idx)+len(training_idx))==Inputs.shape[0]:
        print('len(test_idx)', len(test_idx))
        print('len(training_idx)', len(training_idx))
        print('len(test_idx)+len(training_idx)', len(test_idx)+len(training_idx))
        print('Inputs.shape[0]', Inputs.shape[0])
        print('Test und Training haben falsche Laenge')
        print('test_idx', test_idx)
    Inputs_train, Inputs_test = Inputs[training_idx,:], Inputs[test_idx,:]
    Targets_train, Targets_test = Targets[training_idx,:], Targets[test_idx,:]


    train_val_splitter = 0.9
    train_train_idx_idx = np.random.choice(np.arange(training_idx.shape[0]), int(training_idx.shape[0]*train_val_splitter), replace=False)
    train_train_idx = training_idx[train_train_idx_idx]
    train_val_idx = training_idx[ np.setdiff1d(  np.arange(training_idx.shape[0]), train_train_idx_idx)]
    if not (len(train_val_idx)+len(train_train_idx))==training_idx.shape[0]:
        print('len(index train_val_idx)',len(np.random.choice(np.arange(training_idx.shape[0]), int(training_idx.shape[0]*train_val_splitter), replace=False)))
        print('len(train_val_idx)', len(train_val_idx))
        print('len np.setdiff1d(  np.arange(training_idx.shape[0]), train_train_idx)', len(np.setdiff1d(  np.arange(training_idx.shape[0]), train_train_idx)))
        print('len(train_train_idx)', len(train_train_idx))
        print('len(train_val_idx)+len(train_train_idx)', len(train_val_idx)+len(train_train_idx))
        print('training_idx.shape[0]', training_idx.shape[0])
        print('Validation und Training haben falsche Laenge')
    Inputs_train_train, Inputs_train_val = Inputs[train_train_idx,:], Inputs[train_val_idx,:]
    Targets_train, Targets_test = Targets[train_train_idx,:], Targets[train_val_idx,:]
    if reweighting:
        weights_train_, weights_val_ = Weights[train_train_idx,:], Weights[train_val_idx,:]
    else:
        print("No reweighting")
        weights_train_, weights_val_ = np.repeat(1., len(train_train_idx)) , np.repeat(1., len(train_val_idx))
        weights_train_.shape = (len(train_train_idx),1)
        weights_val_.shape = (len(train_val_idx),1)

    data_train = Inputs_train_train
    labels_train = Targets_train
    data_val = Inputs_train_val
    labels_val = Targets_test
    weights_train = weights_train_
    weights_val = weights_val_
    batchsize = 300
    batchsize_val = 10000
    print("Validation set hat Groesse ", len(train_val_idx))
    x = tf.placeholder(tf.float32, shape=[batchsize, data_train.shape[1]])
    y = tf.placeholder(tf.float32, shape=[batchsize, labels_train.shape[1]], )
    w = tf.placeholder(tf.float32, shape=[batchsize, weights_train.shape[1]])
    x_ = tf.placeholder(tf.float32)
    y_ = tf.placeholder(tf.float32)
    w_ = tf.placeholder(tf.float32)
    #enqueue_train = queue_train.enqueue_many([x, y, w])
    #enqueue_val = queue_val.enqueue_many([x, y, w])

    print("tf.test.gpu_device_name()", tf.test.gpu_device_name())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # ## Define the neural network architecture
    batch_train = [data_train, labels_train, weights_train]
    batch_val = [data_val, labels_val, weights_val]

    print('wichtig',data_train.shape[0], data_train.shape[1])

    logits_train, f_train= NNmodel(x, reuse=False)
    logits_val, f_val= NNmodel(x_, reuse=True)

    # ## Add training operations to the graph
    print('len logits_train', logits_train.shape)

    print("loss fct", loss_fct)
    if (loss_fct=="mean_squared_error"):
        loss_train = tf.reduce_mean(
            tf.losses.mean_squared_error(labels=y, predictions=logits_train))
        loss_val = tf.reduce_mean(
            tf.losses.mean_squared_error(labels=y_, predictions=logits_val))
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    elif (loss_fct=="Response"):
        print("Loss Function Response: ", loss_fct)
        loss_train = costResponse(y, logits_train, w)
        loss_val = costResponse(y_, logits_val, w_)
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    elif (loss_fct=="Resolution_para"):
        print("Loss Function Resolution_para: ", loss_fct)
        loss_train = costResolution_para(batch_train[1], logits_train, batch_train[2])
        loss_val = costResolution_para(batch_val[1], logits_val, batch_val[2])
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    elif (loss_fct=="Resolution_perp"):
        print("Loss Function Resolution_perp: ", loss_fct)
        loss_train = costResolution_perp(batch_train[1], logits_train, batch_train[2])
        loss_val = costResolution_perp(batch_val[1], logits_val, batch_val[2])
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    elif (loss_fct=="Angle_Response"):
        print("Loss Function Angle_Response: ", loss_fct)
        loss_train = costExpected(y, logits_train, w)
        loss_val = costExpected(y_, logits_val, w_)
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    elif (loss_fct=="Angle_ResponsePz"):
        print("Loss Function Angle_ResponsePz: ", loss_fct)
        loss_train = costExpectedPz(y, logits_train, w)
        loss_val = costExpectedPz(y_, logits_val, w_)
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    elif (loss_fct=="Angle_10Response"):
        print("Loss Function 10Angle_Response: ", loss_fct)
        loss_train = cost10Expected(y, logits_train, w)
        loss_val = cost10Expected(y_, logits_val, w_)
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    elif (loss_fct=="relResponse"):
        print("Loss Function Angle_Response rel: ", loss_fct)
        loss_train = costExpectedRel(y, logits_train, w)
        loss_val = costExpectedRel(y_, logits_val, w_)
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    elif (loss_fct=="relResponseAsy"):
        print("Loss Function Angle_Response rel: ", loss_fct)
        loss_train = costExpectedRelAsy(y, logits_train, w)
        loss_val = costExpectedRelAsy(y_, logits_val, w_)
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    elif (loss_fct=="relResponseAsy2"):
        print("Loss Function Angle_Response rel: ", loss_fct)
        loss_train = costExpectedRelAsy2(y, logits_train, w)
        loss_val = costExpectedRelAsy2(y_, logits_val, w_)
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    elif (loss_fct=="Angle_relabsResponse"):
        print("Loss Function Angle_Response rel abs: ", loss_fct)
        loss_train = costExpectedRelAbs(y, logits_train, w)
        loss_val = costExpectedRelAbs(y_, logits_val, w_)
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    elif (loss_fct=="Resolutions"):
        print("Loss Function Resolutions: ", loss_fct)
        loss_train = costResolutions(y, logits_train, w)
        loss_val = costResolutions(y_, logits_val, w_)
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    elif (loss_fct=="MSE"):
        print("Loss Function MSE: ", loss_fct)
        loss_train = costMSE(y, logits_train, w)
        loss_val = costMSE(y_, logits_val, w_)
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    else:
        factor_response, factor_res_para, factor_res_perp, factor_mse = 1,1,1,1
        loss_response = costResponse(batch_train[1], logits_train, batch_train[2])
        loss_res_para = costResolution_para(batch_train[1], logits_train, batch_train[2])
        loss_res_perp = costResolution_perp(batch_train[1], logits_train, batch_train[2])
        loss_MSE = costMSE(batch_train[1], logits_train, batch_train[2])
        loss_final = factor_response * loss_response + factor_res_para * loss_res_para + factor_res_perp * loss_res_perp + factor_mse * loss_MSE
        train_op = tf.optimizer.Adam().minimize(loss_final)



    # ## Run the training
    sess.run(tf.global_variables_initializer())

    losses_train = []
    losses_val = []
    min_valloss = [1000000000000]
    loss_response, loss_resolution_para, loss_resolution_perp, loss_mse = [], [], [], []
    summary_train = tf.summary.scalar("loss_train", loss_train)
    summary_val = tf.summary.scalar("loss_val", loss_val)
    writer = tf.summary.FileWriter("./logs/{}".format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), sess.graph)
    saver = tf.train.Saver()
    saveStep = 100
    early_stopping = 0
    best_model = 0
    print("StartTraining")
    batch_prob = weights_train.flatten() * 1 / (np.sum(weights_train.flatten()))
    print(" shape batch_prob", batch_prob.shape)
    print(" shape data_train", data_train.shape)
    batch_prob_val = weights_val.flatten() * 1 / (np.sum( weights_val.flatten()))
    pT = np.sqrt(np.square(labels_train[:,0]) + np.square(labels_train[:,1]))
    for i_step in range(30000):
        start_loop = time.time()

        #p = map(lambda x: abs(np.random.uniform(min(pT),max(pT)) - x), pT)
        #print(p)
        #batch_train_idx = p.index(min(p))[:batchsize]
        batch_train_idx = np.random.choice(np.arange(data_train.shape[0]), batchsize, p=batch_prob, replace=False)
        #pval = map(lambda x: abs(np.random.uniform(min(pT),max(pT)) - x), pT)
        #batch_val_idx = pval.index(min(pval))[:batchsize]
        batch_val_idx = np.random.choice(np.arange(data_val.shape[0]), batchsize, p=batch_prob_val, replace=False)

        summary_, loss_, _ = sess.run([summary_train, loss_train, minimize_loss], feed_dict={x: data_train[batch_train_idx,:], y: labels_train[batch_train_idx,:], w: weights_train[batch_train_idx,:]})
        #summary_, loss_, loss_response_, loss_resolution_para_, loss_resolution_perp_, loss_mse_, _ = sess.run([summary_train, loss_train, minimize_loss])
        losses_train.append(loss_)
        writer.add_summary(summary_, i_step)
        summary_, loss_ = sess.run([summary_val, loss_val], feed_dict={x_: data_val[batch_val_idx,:], y_: labels_val[batch_val_idx,:], w_: weights_val[batch_val_idx,:]})
        losses_val.append(loss_)
        writer.add_summary(summary_, i_step)
        end_loop = time.time()



        if i_step % saveStep == 0:
            #pval_100 = map(lambda x: abs(np.random.uniform(min(pT),max(pT)) - x), pT)
            #batch_val_idx_100 = pval_100.index(min(pval))[:batchsize_val]
            batch_val_idx_100 =  np.random.choice(np.arange(data_val.shape[0]), batchsize_val, p=batch_prob_val, replace=False)
            loss_ = sess.run(loss_val, feed_dict={x_: data_val[batch_val_idx_100,:], y_: labels_val[batch_val_idx_100,:], w_: weights_val[batch_val_idx_100,:]})
            if loss_<min(min_valloss):
                best_model = i_step
                saver.save(sess, "%sCV/NNmodel_CV"%outputDir, global_step=i_step)
                outputs = ["output"] # names of output operations you want to use later
                constant_graph = tf.graph_util.convert_variables_to_constants(
                    sess, sess.graph.as_graph_def(), outputs)
                tf.train.write_graph(constant_graph, outputDir, "constantgraph.pb", as_text=False)

                early_stopping = 0
                pT2 = np.sqrt(np.square(labels_train[batch_train_idx,0]) + np.square(labels_train[batch_train_idx,1]))
                print("better val loss found at ", i_step)

            else:
                early_stopping += 1
                print("increased early stopping to ", early_stopping)
            if early_stopping == 15:
                break
            min_valloss.append(loss_)
            print('gradient step No ', i_step)
            print("validation loss", loss_)
            print("gradient step time {0} seconds".format(end_loop-start_loop))






    #writer.flush()
    pT_woutWeight = np.sqrt(np.square(labels_train[:,0])+np.square(labels_train[:,1]))
    #plt.hist(pT_woutWeight, bins=18, lw=3, label="train pT distr")
    plt.hist(pT_woutWeight[np.random.choice(np.arange(data_train.shape[0]), batchsize, replace=True)], bins=18, lw=3, label="Input distribution", histtype="step")
    plt.hist(pT2, bins=18, lw=3, label="weighted random choice", histtype="step")

    plt.xlabel("$p_T^Z$"), plt.ylabel("Count")
    plt.legend()
    plt.savefig("%sBatch_CV.png"%(plotsD))
    plt.close()


    plt.plot(range(1, len(moving_average(np.asarray(losses_train[0:(best_model+1500)]), 1500))+1), moving_average(np.asarray(losses_train[0:(best_model+1500)]), 1500), lw=3, label="Training loss")
    plt.plot(range(1, len(moving_average(np.asarray(losses_val[0:(best_model+1500)]), 1500))+1), moving_average(np.asarray(losses_val[0:(best_model+1500)]), 1500), lw=3, label="Validation loss")
    plt.xlabel("Gradient step"), plt.ylabel("loss")
    plt.yscale('log')
    plt.legend()
    plt.savefig("%sLoss_ValLoss_CV.png"%(plotsD))
    plt.close()



    if loss_fct=="all":
        plt.plot(range(1, len(moving_average(np.asarray(loss_response), 800))+1), moving_average(np.asarray(losses_response), 800), lw=1.5, label="Response loss")
        plt.plot(range(1, len(moving_average(np.asarray(loss_res_para), 800))+1), moving_average(np.asarray(loss_res_para), 800), lw=1.5, label="Resolution para loss")
        plt.plot(range(1, len(moving_average(np.asarray(loss_res_perp), 800))+1), moving_average(np.asarray(loss_res_perp), 800), lw=1.5, label="Resolution perp loss")
        plt.plot(range(1, len(moving_average(np.asarray(loss_mse), 800))+1), moving_average(np.asarray(loss_mse), 800), lw=1.5, label="MSE loss")
        plt.plot(range(1, len(moving_average(np.asarray(losses_train), 800))+1), moving_average(np.asarray(losses_train), 800), lw=3, label="loss")
        plt.xlabel("Gradient step"), plt.ylabel("loss")
        plt.legend()
        plt.savefig("%sLosses.png"%(plotsD))
        plt.close()




    dset = NN_Output.create_dataset("loss", dtype='f', data=losses_train)
    dset2 = NN_Output.create_dataset("val_loss", dtype='f', data=losses_val)
    NN_Output.close()

    end = time.time()
    print("program needed {0} seconds".format(end-start))

if __name__ == "__main__":
    outputDir = sys.argv[1]
    optim = str(sys.argv[2])
    loss_fct = str(sys.argv[3])
    NN_mode = sys.argv[4]
    plotsD = sys.argv[5]
    print(outputDir)
    NN_Output = h5py.File("%sNN_Output_%s.h5"%(outputDir,NN_mode), "w")
    Test_Idx = h5py.File("%sTest_Idx_CV_%s.h5" % (outputDir, NN_mode), "w")
    getModel(outputDir, optim, loss_fct, NN_mode, plotsD)
