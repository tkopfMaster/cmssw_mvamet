import numpy as np
np.random.seed(1234)
import tensorflow as tf
import datetime
import sys
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D as plt3d


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


def getModel(outputDir, optim, loss_fct, NN_mode, plotsD):
    Inputs, Targets = loadInputsTargets(outputDir)
    num_events = Inputs.shape[0]
    print('Number of events in get model ', num_events)
    train_test_splitter = 0.9
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


    # In[5]:


    sess = tf.Session()
    sess.run(enqueue_train, feed_dict={x: data_train, y: labels_train})
    sess.run(enqueue_val, feed_dict={x: data_val, y: labels_val})


    # ## Define the neural network architecture

    # In[6]:


    def model(x, reuse):
        with tf.variable_scope("model") as scope:
            if reuse:
                scope.reuse_variables()
            w1 = tf.get_variable('w1', shape=(data_train.shape[1],data_train.shape[0]), dtype=tf.float32,
                    initializer=tf.random_normal_initializer(stddev=0.1))
            b1 = tf.get_variable('b1', shape=(data_train.shape[0]), dtype=tf.float32,
                    initializer=tf.constant_initializer(0.1))
            w2 = tf.get_variable('w2', shape=(data_train.shape[0], 2), dtype=tf.float32,
                    initializer=tf.random_normal_initializer(stddev=0.1))
            b2 = tf.get_variable('b2', shape=(2), dtype=tf.float32,
                    initializer=tf.constant_initializer(0.1))

        l1 = tf.nn.sigmoid(tf.add(b1, tf.matmul(x, w1)))
        logits = tf.add(b2, tf.matmul(l1, w2))
        return logits, logits

    batch_size = 32
    batch_train = queue_train.dequeue_many(batch_size)
    batch_val = queue_val.dequeue_many(batch_size)

    logits_train, f_train = model(batch_train[0], reuse=False)
    logits_val, f_val = model(batch_val[0], reuse=True)

    # ## Add training operations to the graph
    print('len len(batch_train[1])', batch_train[1])
    print('len logits_train', logits_train.shape)
    loss_train = tf.reduce_mean(
        tf.losses.mean_squared_error(labels=batch_train[1], predictions=logits_train))
    minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)

    loss_val = tf.reduce_mean(
        tf.losses.mean_squared_error(labels=batch_val[1], predictions=logits_val))


    # ## Run the training

    # In[8]:


    sess.run(tf.global_variables_initializer())

    losses_train = []
    losses_val = []

    summary_train = tf.summary.scalar("loss_train", loss_train)
    summary_val = tf.summary.scalar("loss_val", loss_val)
    writer = tf.summary.FileWriter("./logs/{}".format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), sess.graph)

    for i_step in range(100):
        summary_, loss_, _ = sess.run([summary_train, loss_train, minimize_loss])
        losses_train.append(loss_)
        writer.add_summary(summary_, i_step)

        summary_, loss_ = sess.run([summary_val, loss_val])
        losses_val.append(loss_)
        writer.add_summary(summary_, i_step)

    writer.flush()


    # In[9]:


    plt.plot(range(1, len(losses_train)+1), losses_train, lw=3, label="Training loss")
    plt.plot(range(1, len(losses_val)+1), losses_val, lw=3, label="Validation loss")
    plt.xlabel("Gradient step"), plt.ylabel("Cross-entropy loss")
    plt.legend()
    plt.savefig("%sLoss_ValLoss.png"%(plotsD))
    plt.close()


    dset = NN_Output.create_dataset("loss", dtype='f', data=losses_train)
    dset2 = NN_Output.create_dataset("val_loss", dtype='f', data=losses_val)
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
