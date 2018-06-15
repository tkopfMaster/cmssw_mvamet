from keras import backend as K
import numpy as np



def mean_squared_error_r(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true)+10*K.abs(K.square(y_pred)-K.square(y_true)), axis=-1)

def perp_long_error(y_true, y_pred):
    raise Exception("DEBUG:", y_true.shape, y_pred.shape)
    a_=K-sqrt(K.square(y_pred))
    pZ = K.sqrt(K.square(y_true))
    alpha_a=K.arctan2(K.divide(y_pred[1],y_pred[0]))
    alpha_Z=K.arctan2(K.divide(y_true[1],y_true[0]))
    alpha_diff=alpha_a-alpha_Z
    u_perp = K.sin(alpha_diff)*a_
    u_long = K.cos(alpha_diff)*a_

    return K.mean(K.square(u_perp)+K.square(u_long-pZ), axis=-1)
