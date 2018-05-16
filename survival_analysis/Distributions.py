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



    def left_censoring(self, time, Lambda):
        return self.integrated(tf.constant(0.0), time, Lambda)

    def right_censoring(self, time, Lambda):
        return self.integrated(time, tf.constant(np.inf), Lambda)

    def integrated(self, lower, upper, Lambda):
        return tf.exp(-1 * Lambda * lower ** self.shape) - tf.exp(-1 * Lambda * upper ** self.shape)
