import numpy as np
np.random.seed(1234)
import tensorflow as tf
import datetime
import sys
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D as plt3d
import time



def loadInputsTargets(outputD):
    InputsTargets = h5py.File("%sNN_Input_training_%s.h5" % (outputD,NN_mode), "r")
    norm = np.sqrt(np.multiply(InputsTargets['Target'][:,0],InputsTargets['Target'][:,0]) + np.multiply(InputsTargets['Target'][:,1],InputsTargets['Target'][:,1]))

    Target =  InputsTargets['Target']
    Input = np.row_stack((
                InputsTargets['PF'],
                InputsTargets['Track'],
                InputsTargets['NoPU'],
                InputsTargets['PUCorrected'],
                InputsTargets['PU'],
                InputsTargets['Puppi']
                ))
    return (np.transpose(Input), np.transpose(Target))

def Names(outputD):
    InputsTargets = h5py.File("%sNN_Input_training_%s.h5" % (outputD,NN_mode), "r")
    return (InputsTargets.keys)

def costResponseResolution(y_true,y_pred):
    a_=tf.sqrt(tf.square(y_pred[:,0])+tf.square(y_pred[:,1]))
    pZ = tf.sqrt(tf.square(y_true[:,0])+tf.square(y_true[:,1]))
    alpha_a=tf.atan2(y_pred[:,1],y_pred[:,0])
    alpha_Z=tf.atan2(y_true[:,1],y_true[:,0])
    alpha_diff=tf.subtract(alpha_a,alpha_Z)
    u_perp = tf.sin(alpha_diff)*a_
    u_long = tf.cos(alpha_diff)*a_

    cost = tf.square(u_perp)+tf.square(tf.add(u_long,-pZ))
    #cost = tf.square(np.divide(u_perp,pZ))+tf.square(tf.divide(u_long,pZ)-1)
    return tf.reduce_mean(cost)

def NNmodel(x, reuse):
    with tf.variable_scope("model") as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable('w1', shape=(12,100), dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1', shape=(100), dtype=tf.float32,
                initializer=tf.constant_initializer(0.1))
        w2 = tf.get_variable('w2', shape=(100, 2), dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2', shape=(2), dtype=tf.float32,
                initializer=tf.constant_initializer(0.1))

    l1 = tf.nn.sigmoid(tf.add(b1, tf.matmul(x, w1)))
    logits = tf.add(b2, tf.matmul(l1, w2), name='logits')
    return logits, logits


def getModel(outputDir, optim, loss_fct, NN_mode, plotsD):
    start = time.time()
    Inputs, Targets = loadInputsTargets(outputDir)
    num_events = Inputs.shape[0]
    print('Number of events in get model ', num_events)
    train_test_splitter = 0.1
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



    data_train = Inputs_train_train
    labels_train = Targets_train

    data_val = Inputs_train_val
    labels_val = Targets_test


    # ## Load data to a queue
    print('length data_train.shape[1]', data_train.shape[1])
    print('length labels_train.shape[1]', labels_train.shape[1])
    print('length data_train.shape[1], labels_train.shape[1]', [data_train.shape[1], labels_train.shape[1]])
    queue_train = tf.RandomShuffleQueue(capacity=data_train.shape[0], min_after_dequeue=0,
                                        dtypes=[tf.float32, tf.float32], shapes=[tf.TensorShape(data_train.shape[1]),tf.TensorShape(labels_train.shape[1])])

    queue_val = tf.RandomShuffleQueue(capacity=data_val.shape[0], min_after_dequeue=0,
                                      dtypes=[tf.float32, tf.float32], shapes=[tf.TensorShape(data_val.shape[1]),tf.TensorShape(labels_val.shape[1])])

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    enqueue_train = queue_train.enqueue_many([x, y])
    enqueue_val = queue_val.enqueue_many([x, y])



    sess = tf.Session()
    sess.run(enqueue_train, feed_dict={x: data_train, y: labels_train})
    sess.run(enqueue_val, feed_dict={x: data_val, y: labels_val})
    #raise Exception("DEBUG")

    # ## Define the neural network architecture


    batch_size = 32

    batch_train = queue_train.dequeue_many(batch_size)
    batch_val = queue_val.dequeue_many(batch_size)
    print('wichtig',data_train.shape[0], data_train.shape[1])

    logits_train, f_train = NNmodel(batch_train[0], reuse=False)
    logits_val, f_val = NNmodel(batch_val[0], reuse=True)

    # ## Add training operations to the graph
    print('len len(batch_train[1])', batch_train[1])
    print('len logits_train', logits_train.shape)


    if (loss_fct=="mean_squared_error"):
        loss_train = tf.reduce_mean(
            tf.losses.mean_squared_error(labels=batch_train[1], predictions=logits_train))
        loss_val = tf.reduce_mean(
            tf.losses.mean_squared_error(labels=batch_val[1], predictions=logits_val))
    else:
        loss_train = costResponseResolution(batch_train[1], logits_train)
        loss_val = costResponseResolution(batch_val[1], logits_val)
    minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)





    # ## Run the training
    sess.run(tf.global_variables_initializer())

    losses_train = []
    losses_val = []

    summary_train = tf.summary.scalar("loss_train", loss_train)
    summary_val = tf.summary.scalar("loss_val", loss_val)
    #writer = tf.summary.FileWriter("./logs/{}".format(
    #        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), sess.graph)
    saver = tf.train.Saver()
    for i_step in range(5000):
        start_loop = time.time()
        summary_, loss_, _ = sess.run([summary_train, loss_train, minimize_loss])
        losses_train.append(loss_)
        ##writer.add_summary(summary_, i_step)

        summary_, loss_ = sess.run([summary_val, loss_val])
        losses_val.append(loss_)
        #writer.add_summary(summary_, i_step)
        #saver.save(sess, "%sNNmodel"%outputDir, global_step=i_step)
        end_loop = time.time()
        if i_step%100==0:
            print("target", sess.run(batch_train[1]), " prediction ", sess.run(logits_train))
            print('gradient step No ', i_step)
            print("gradient step time {0} seconds".format(end_loop-start_loop))


    #writer.flush()



    plt.plot(range(1, len(losses_train)+1), losses_train, lw=3, label="Training loss")
    plt.plot(range(1, len(losses_val)+1), losses_val, lw=3, label="Validation loss")
    plt.xlabel("Gradient step"), plt.ylabel("loss")
    plt.legend()
    #plt.savefig("%sLoss_ValLoss.png"%(plotsD))
    plt.close()


    #dset = NN_Output.create_dataset("loss", dtype='f', data=losses_train)
    #dset2 = NN_Output.create_dataset("val_loss", dtype='f', data=losses_val)
    #NN_Output.close()
    end = time.time()
    print("program needed {0} seconds".format(end-start))

if __name__ == "__main__":
    outputDir = sys.argv[1]
    optim = str(sys.argv[2])
    loss_fct = str(sys.argv[3])
    NN_mode = sys.argv[4]
    plotsD = sys.argv[5]
    print(outputDir)
    #NN_Output = h5py.File("%sNN_Output_%s.h5"%(outputDir,NN_mode), "w")
    getModel(outputDir, optim, loss_fct, NN_mode, plotsD)
