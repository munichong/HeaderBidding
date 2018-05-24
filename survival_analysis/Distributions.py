import numpy as np
import tensorflow as tf
import scipy.integrate as integrate

class WeibullDistribution:

    def __init__(self, shape=0.4, threshold=0.05):
        '''

        :param shape: The Weibull distribution with shape = 1 is the exponential distribution
                      A value of shape > 1 indicates that the failure rate increases over time.
        :param threshold:
        '''
        self.shape = shape
        self.threshold = threshold

    def left_censoring(self, time, Lambda):
        return 1 - self.right_censoring(time, Lambda)

    def right_censoring(self, time, Lambda):
        return tf.exp(-1 * Lambda * time ** self.shape)
