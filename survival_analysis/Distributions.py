import numpy as np
import tensorflow as tf
import scipy.integrate as integrate

class WeibullDistribution:

    def __init__(self, shape=1, threshold=0.05):
        '''

        :param shape: The Weibull distribution with shape = 1 is the exponential distribution
        :param threshold:
        '''
        self.shape = shape
        self.threshold = threshold

    def f_t(self, time, Lambda):
        h = Lambda * self.shape * time ** (self.shape - 1)
        S = tf.exp(-1 * Lambda * time ** self.shape)
        return h * S


    def left_censoring(self, time, Lambda):
        time = tf.expand_dims(time, axis=-1)
        return tf.map_fn(lambda y: tf.py_func(
            lambda x: integrate.quad(self.f_t, 0.0, x[0],args=(x[1],))[0], [y], tf.float64
        ), tf.concat([time, Lambda], 1))

    def right_censoring(self, time, Lambda):
        time = tf.expand_dims(time, axis=-1)
        return tf.map_fn(lambda y: tf.py_func(
            lambda x: integrate.quad(self.f_t, x[0], np.inf, args=(x[1],))[0], [y], tf.float64
        ), tf.concat([time, Lambda], 1))

