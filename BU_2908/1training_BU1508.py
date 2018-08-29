import numpy as np
np.random.seed(1234)
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
import datetime
import sys
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D as plt3d
import time



def loadInputsTargetsWeights(outputD):
    InputsTargets = h5py.File("%sNN_Input_training_%s.h5" % (outputD,NN_mode), "r")
    norm = np.sqrt(np.multiply(InputsTargets['Target'][:,0],InputsTargets['Target'][:,0]) + np.multiply(InputsTargets['Target'][:,1],InputsTargets['Target'][:,1]))

    Target =  InputsTargets['Target']
    weight =  InputsTargets['weights']
    Input = np.row_stack((
                InputsTargets['PF'],
                InputsTargets['Track'],
                InputsTargets['NoPU'],
                InputsTargets['PUCorrected'],
                InputsTargets['PU'],
                InputsTargets['Puppi']
                ))
    return (np.transpose(Input), np.transpose(Target), np.transpose(weight))

def moving_average(data_set, periods):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')

def Names(outputD):
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

    cost= tf.multiply(u_perp_, tf.square(weight))
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

    cost = tf.multiply(tf.square(Resolution_para), tf.square(weight))
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

    cost = tf.multiply(tf.square(Resolution_para)+tf.square(u_perp_), tf.square(weight))
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

    cost = tf.multiply(tf.square(Response-1)+tf.square(tf.sin(alpha_diff)), tf.square(weight))
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

    cost = tf.multiply(tf.square(Resolution_para)+tf.square(u_perp), tf.square(weight))
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

    cost= tf.multiply(tf.square(tf.multiply(Response-1,pZ)), tf.square(weight))
    return tf.reduce_mean(cost)


def costMSE(y_true,y_pred, weight):
    MSE_x=tf.square(y_pred[:,0]-y_true[:,0])
    MSE_y = tf.square(y_pred[:,1]-y_true[:,1])

    cost = tf.multiply(MSE_x+MSE_y, tf.square(weight))
    return tf.reduce_mean(cost)


def NNmodel(x, reuse):
    with tf.variable_scope("model") as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable('w1', shape=(12,240), dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1', shape=(240), dtype=tf.float32,
                initializer=tf.constant_initializer(0.1))
        w2 = tf.get_variable('w2', shape=(240, 240), dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2', shape=(240), dtype=tf.float32,
                initializer=tf.constant_initializer(0.1))
        w3 = tf.get_variable('w3', shape=(240, 240), dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=0.1))
        b3 = tf.get_variable('b3', shape=(240), dtype=tf.float32,
                initializer=tf.constant_initializer(0.1))
        w4 = tf.get_variable('w4', shape=(240, 240), dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=0.1))
        b4 = tf.get_variable('b4', shape=(240), dtype=tf.float32,
                initializer=tf.constant_initializer(0.1))
        w5 = tf.get_variable('w5', shape=(240, 2), dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=0.1))
        b5 = tf.get_variable('b5', shape=(2), dtype=tf.float32,
                initializer=tf.constant_initializer(0.1))

    l1 = tf.nn.sigmoid(tf.add(b1, tf.matmul(x, w1)))
    l2 = tf.nn.sigmoid(tf.add(b2, tf.matmul(l1, w2)))
    l3 = tf.nn.sigmoid(tf.add(b3, tf.matmul(l2, w3)))
    l4 = tf.nn.sigmoid(tf.add(b4, tf.matmul(l3, w4)))
    logits = tf.add(b5, tf.matmul(l4, w5), name='logits')
    return logits, logits


def getModel(outputDir, optim, loss_fct, NN_mode, plotsD):
    start = time.time()
    Inputs, Targets, Weights = loadInputsTargetsWeights(outputDir)
    Boson_Pt = np.sqrt(np.square(Targets[:,0])+np.square(Targets[:,1]))
    num_events = Inputs.shape[0]
    print('Number of events in get model ', num_events)
    train_test_splitter = 0.3
    training_idx = np.random.choice(np.arange(Inputs.shape[0]), int(Inputs.shape[0]*train_test_splitter), replace=False)
    print('random training index length', training_idx.shape)
    print('inputs shape', Inputs.shape)
    test_idx = np.setdiff1d(  np.arange(Inputs.shape[0]), training_idx)
    if not (len(test_idx)+len(training_idx))==Inputs.shape[0]:
        print('len(test_idx)', len(test_idx))
        print('len(training_idx)', len(training_idx))
        print('len(test_idx)+len(training_idx)', len(test_idx)+len(training_idx))
        print('Inputs.shape[0]', Inputs.shape[0])
        print('Test und Training haben falsche Laenge')
        print('test_idx', test_idx)
    Inputs_train, Inputs_test = Inputs[training_idx,:], Inputs[test_idx,:]
    Targets_train, Targets_test = Targets[training_idx,:], Targets[test_idx,:]


    train_val_splitter = 0.5
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
    weights_train_, weights_val_ = Weights[train_train_idx,:], Weights[train_val_idx,:]

    data_train = Inputs_train_train
    labels_train = Targets_train
    data_val = Inputs_train_val
    labels_val = Targets_test
    weights_train = weights_train_
    weights_val = weights_val_
    '''
    for i in range(1000):
        data_train = np.vstack((data_train, Inputs_train_train))
        labels_train = np.vstack((labels_train, Targets_train))
        weights_train = np.vstack((weights_train, weights_train_))
        data_val = np.vstack((data_val,Inputs_train_val))
        labels_val = np.vstack((labels_val, Targets_test ))
        weights_val = np.vstack((weights_val,weights_val_))
    '''



    # ## Load data to a queue
    print('length data_train.shape[1]', data_train.shape[1])
    print('length labels_train.shape[1]', labels_train.shape[1])
    print('length data_train.shape[1], labels_train.shape[1]', [data_train.shape[1], labels_train.shape[1]])

    queue_train = tf.RandomShuffleQueue(capacity=data_train.shape[0], min_after_dequeue=0,
                                        dtypes=[tf.float32, tf.float32, tf.float32], shapes=[tf.TensorShape(data_train.shape[1]),tf.TensorShape(labels_train.shape[1]),tf.TensorShape(weights_train.shape[1])])

    queue_val = tf.RandomShuffleQueue(capacity=data_val.shape[0], min_after_dequeue=0,
                                      dtypes=[tf.float32, tf.float32, tf.float32], shapes=[tf.TensorShape(data_val.shape[1]),tf.TensorShape(labels_val.shape[1]),tf.TensorShape(weights_val.shape[1])])

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    w = tf.placeholder(tf.float32)
    batch_size = 30000
    enqueue_train = queue_train.enqueue_many([x, y, w])
    enqueue_val = queue_val.enqueue_many([x, y, w])
    qtrain = tf.FIFOQueue(capacity=200000, dtypes=[tf.float32, tf.float32, tf.float32], shapes=[tf.TensorShape(data_val.shape[1]),tf.TensorShape(labels_val.shape[1]),tf.TensorShape(weights_val.shape[1])])
    sess = tf.Session()
    enqueue_op_train = qtrain.enqueue_many([x, y, w])
    qval = tf.FIFOQueue(capacity=200000,dtypes=[tf.float32, tf.float32, tf.float32], shapes=[tf.TensorShape(data_val.shape[1]),tf.TensorShape(labels_val.shape[1]),tf.TensorShape(weights_val.shape[1])])
    enqueue_op_val = qval.enqueue_many([x, y, w])

    sess.run(enqueue_train, feed_dict={x: data_train, y: labels_train, w: weights_train})
    sess.run(enqueue_val, feed_dict={x: data_val, y: labels_val, w: weights_val})

    queue_runner_train = tf.train.QueueRunner(qtrain, [enqueue_op_train])
    tf.train.add_queue_runner(queue_runner_train)
    queue_runner_val = tf.train.QueueRunner(qval, [enqueue_op_val])
    tf.train.add_queue_runner(queue_runner_val)
    coordinator = tf.train.Coordinator()
    threads_train = queue_runner_train.create_threads(sess, coord=coordinator, start=True)
    threads_val = queue_runner_val.create_threads(sess, coord=coordinator, start=True)
    # ## Define the neural network architecture


    batch_train = qtrain.dequeue_many(batch_size)
    batch_val = qval.dequeue_many(batch_size)
    print('wichtig',data_train.shape[0], data_train.shape[1])


    logits_train, f_train= NNmodel(batch_train[0], reuse=False)
    logits_val, f_val= NNmodel(batch_val[0], reuse=True)

    # ## Add training operations to the graph
    print('len len(batch_train[1])', batch_train[1])
    print('len logits_train', logits_train.shape)

    print("loss fct", loss_fct)
    if (loss_fct=="mean_squared_error"):
        loss_train = tf.reduce_mean(
            tf.losses.mean_squared_error(labels=batch_train[1], predictions=logits_train))
        loss_val = tf.reduce_mean(
            tf.losses.mean_squared_error(labels=batch_val[1], predictions=logits_val))
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    elif (loss_fct=="Response"):
        print("Loss Function Response: ", loss_fct)
        loss_train = costResponse(batch_train[1], logits_train, batch_train[2])
        loss_val = costResponse(batch_val[1], logits_val, batch_val[2])
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
        loss_train = costExpected(batch_train[1], logits_train, batch_train[2])
        loss_val = costExpected(batch_val[1], logits_val, batch_val[2])
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    elif (loss_fct=="Angle_relResponse"):
        print("Loss Function Angle_Response rel: ", loss_fct)
        loss_train = costExpectedRel(batch_train[1], logits_train, batch_train[2])
        loss_val = costExpectedRel(batch_val[1], logits_val, batch_val[2])
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    elif (loss_fct=="Resolutions"):
        print("Loss Function Angle_Response: ", loss_fct)
        loss_train = costResolutions(batch_train[1], logits_train, batch_train[2])
        loss_val = costResolutions(batch_val[1], logits_val, batch_val[2])
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
    sess.run(tf.local_variables_initializer())

    losses_train = []
    losses_val = []
    loss_response, loss_resolution_para, loss_resolution_perp, loss_mse = [], [], [], []

    summary_train = tf.summary.scalar("loss_train", loss_train)
    summary_val = tf.summary.scalar("loss_val", loss_val)
    writer = tf.summary.FileWriter("./logs/{}".format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), sess.graph)
    saver = tf.train.Saver()
    saveStep = 1000
    print("training started")
    for i_step in range(10000):
        start_loop = time.time()

        summary_, loss_, _ = sess.run([summary_train, loss_train, minimize_loss])
        #summary_, loss_, _ = sess.run([summary_train, loss_train, minimize_loss], feed_dict={x: data_train, y: labels_train, w: weights_train})
        #summary_, loss_, loss_response_, loss_resolution_para_, loss_resolution_perp_, loss_mse_, _ = sess.run([summary_train, loss_train, minimize_loss])
        losses_train.append(loss_)
        #loss_response.append(loss_response_)
        #loss_resolution_para.append(loss_resolution_para_)
        #loss_resolution_perp.append(loss_resolution_perp_)
        #loss_mse.append(loss_mse_)
        writer.add_summary(summary_, i_step)

        summary_, loss_ = sess.run([summary_val, loss_val], feed_dict={x: data_val, y: labels_val, w: weights_val})
        losses_val.append(loss_)
        writer.add_summary(summary_, i_step)
        end_loop = time.time()
        if i_step % saveStep == 0:
            saver.save(sess, "%sNNmodel"%outputDir, global_step=i_step)
            print('gradient step No ', i_step)
            print("gradient step time {0} seconds".format(end_loop-start_loop))

    coordinator.request_stop()
    coordinator.join(threads_train)
    coordinator.join(threads_val)




    #writer.flush()


    plt.plot(range(1, len(moving_average(np.asarray(losses_train), 800))+1), moving_average(np.asarray(losses_train), 800), lw=3, label="Training loss")
    plt.plot(range(1, len(moving_average(np.asarray(losses_val), 800))+1), moving_average(np.asarray(losses_val), 800), lw=3, label="Validation loss")
    plt.xlabel("Gradient step"), plt.ylabel("loss")
    plt.legend()
    plt.savefig("%sLoss_ValLoss.png"%(plotsD))
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

    acc, acc_op = tf.metrics.accuracy(labels=logits_train,
                                  predictions=batch_train[1],
                                  weights=batch_train[2])

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
    getModel(outputDir, optim, loss_fct, NN_mode, plotsD)
