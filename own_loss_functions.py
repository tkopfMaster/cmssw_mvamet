from keras import backend as K



def mean_squared_error_r(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true)+K.abs(K.square(y_pred)-K.square(y_true)), axis=-1)
