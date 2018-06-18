from keras import backend as K
import numpy as np
import tensorflow as tf


def mean_squared_error_r(y_true, y_pred):
    return K.mean(K.square(y_pred- y_true)+10*K.abs(K.square(y_pred)-K.square(y_true)), axis=-1)

def perp_long_error(y_true, y_pred):
    a_=K.sqrt(K.square(y_pred[:,0])+K.square(y_pred[:,1]))
    pZ = K.sqrt(K.square(y_true[:,0])+K.square(y_true[:,1]))
    alpha_a=tf.atan2(y_pred[:,1],y_pred[:,0])
    alpha_Z=tf.atan2(y_true[:,1],y_true[:,0])
    alpha_diff=tf.subtract(alpha_a,alpha_Z)
    u_perp = K.sin(alpha_diff)*a_
    u_long = K.cos(alpha_diff)*a_

    return K.square(u_perp)+K.square(tf.subtract(u_long,pZ))

def debug(y_true, y_pred):
    y_pred[np.isnan(y_pred)]=0
    a_=(tf.square(y_pred[:,0])+tf.square(y_pred[:,1]))
    pZ = (tf.square(y_true[:,0])+tf.square(y_true[:,1]))
    y = tf.convert_to_tensor(y_pred[:,1])
    x = tf.convert_to_tensor(y_pred[:,0])
    alpha_a=tf.atan2(y_pred[:,1],y_pred[:,0])
    alpha_Z=tf.atan2(y_true[:,1],y_true[:,0])
    alpha_diff=tf.subtract(alpha_a,alpha_Z)
    u_perp = tf.sin(alpha_diff)*tf.sin(alpha_diff)*a_
    u_long = tf.cos(alpha_diff)*tf.cos(alpha_diff)*a_
    return  tf.atan2(y,x)
