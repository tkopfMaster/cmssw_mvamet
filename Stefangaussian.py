
# coding: utf-8

# # Training a neural network on a binary-classification task using TensorFlow

# In[1]:


import numpy as np
np.random.seed(1234)
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D as plt3d
get_ipython().magic(u'matplotlib inline')


# ## Create the dataset

# In[2]:


num_events = 10000

signal_mean = [1.0, 1.0]
signal_cov = [[1.0, 0.0],
              [0.0, 1.0]]
signal_train = np.random.multivariate_normal(
        signal_mean, signal_cov, num_events)
signal_val = np.random.multivariate_normal(
        signal_mean, signal_cov, num_events)

background_mean = [-1.0, -1.0]
background_cov = [[1.0, 0.0],
                  [0.0, 1.0]]
background_train = np.random.multivariate_normal(
        background_mean, background_cov, num_events)
background_val = np.random.multivariate_normal(
        background_mean, background_cov, num_events)

data_train = np.vstack([signal_train, background_train])
labels_train = np.vstack([np.ones((num_events, 1)), np.zeros((num_events, 1))])

data_val = np.vstack([signal_val, background_val])
labels_val = np.vstack([np.ones((num_events, 1)), np.zeros((num_events, 1))])


# In[3]:


range_ = ((-3, 3), (-3, 3))
plt.figure(0, figsize=(8,4))
plt.subplot(1,2,1); plt.title("Signal")
plt.xlabel("x"), plt.ylabel("y")
plt.hist2d(signal_train[:,0], signal_train[:,1],
        range=range_, bins=20, cmap=cm.coolwarm)
plt.subplot(1,2,2); plt.title("Background")
plt.hist2d(background_train[:,0], background_train[:,1],
        range=range_, bins=20, cmap=cm.coolwarm)
plt.xlabel("x"), plt.ylabel("y");


# ## Load data to a queue

# In[4]:


queue_train = tf.RandomShuffleQueue(capacity=data_train.shape[0], min_after_dequeue=0,
                                    dtypes=[tf.float32, tf.float32], shapes=[[2],[1]])
queue_val = tf.RandomShuffleQueue(capacity=data_val.shape[0], min_after_dequeue=0,
                                  dtypes=[tf.float32, tf.float32], shapes=[[2],[1]])

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
        w1 = tf.get_variable('w1', shape=(2, 100), dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1', shape=(100), dtype=tf.float32,
                initializer=tf.constant_initializer(0.1))
        w2 = tf.get_variable('w2', shape=(100, 1), dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2', shape=(1), dtype=tf.float32,
                initializer=tf.constant_initializer(0.1))

    l1 = tf.nn.relu(tf.add(b1, tf.matmul(x, w1)))
    logits = tf.add(b2, tf.matmul(l1, w2))
    return logits, tf.sigmoid(logits)

batch_size = 32
batch_train = queue_train.dequeue_many(batch_size)
batch_val = queue_val.dequeue_many(batch_size)

logits_train, f_train = model(batch_train[0], reuse=False)
logits_val, f_val = model(batch_val[0], reuse=True)


# ## Add training operations to the graph

# In[7]:


loss_train = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_train[1], logits=logits_train))
minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)

loss_val = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_val[1], logits=logits_val))


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
plt.legend();
