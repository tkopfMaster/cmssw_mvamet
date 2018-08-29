import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32)
y = tf.get_variable("y", initializer=np.array([[1.0], [3.0]], dtype="float32"))
z = tf.matmul(x, y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_ = np.array([3.0, 4.0])
    z_ = sess.run(z, feed_dict={x: np.array([[2.0, 4.0]], dtype="float32")})
    print("1*2 + 3*4 = {}".format(np.squeeze(z_)))
