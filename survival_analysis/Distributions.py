import numpy as np
import tensorflow as tf


class WeibullDistribution:
    def __init__(self, shape=0.4, threshold=0.05):
        '''

        :param shape: The Weibull distribution with shape = 1 is the exponential distribution
                      A value of shape > 1 indicates that the failure rate increases over time.
        :param threshold:
        '''
        self.shape = shape
        self.threshold = threshold

    def left_censoring(self, time, scale):
        return 1 - self.right_censoring(time, scale)

    def right_censoring(self, time, scale):
        # with time increases, the return value should decrease
        return tf.exp(-1 * scale * time ** self.shape)


class LogLogisticDistribution:
    def __init__(self, shape=0.5):
        self.shape = shape

    def left_censoring(self, time, scale):
        return 1 - self.right_censoring(time, scale)

    def right_censoring(self, time, scale):
        return 1 / (scale * time ** self.shape + 1)


# class GammaDistribution:
#     def __init__(self, shape=1.0):
#         self.shape = shape
#
#     def left_censoring(self, time, scale):
#         return 1- self.right_censoring(time, scale)
#
#     def right_censoring(self, time, scale):
#         return self.shape * tf.igammac(self.shape, scale * time)


class GumbelDistribution:
    def __init__(self, shape=0.001):
        self.shape = shape  # this param is actually called "location" in Statistics

    def double_exp_part(self, time, scale):
        return tf.exp(-1 * tf.exp((self.shape - time) / scale))

    def left_censoring(self, time, scale):
        return self.double_exp_part(time, scale) - self.double_exp_part(0.0, scale)

    def right_censoring(self, time, scale):
        return 1 - self.double_exp_part(time, scale)

class ExponentialDistribution:
    def left_censoring(self, time, scale):
        return 1 - self.right_censoring(time, scale)

    def right_censoring(self, time, scale):
        return tf.exp(-1 * scale * time)

