import numpy as np
import tensorflow as tf

class WeibullDistribution:

    def __init__(self, shape=1, threshold=0.05):
        '''

        :param shape: The Weibull distribution with shape = 1 is the exponential distribution
        :param threshold:
        '''
        self.shape = shape
        self.threshold = threshold

    def gradient(self, time, Lambda):
        '''

        :param time: i.e., lambda, tf.constant
        :param Lambda: regression
        :param shape: i.e., p, tf.constant
        :param threshold: tf.constant
        :return:
        '''
        return tf.where(tf.not_equal(abs(time), np.inf),
                        self.gradient_finite(time, Lambda),
                        self.gradient_infinite(time))

    def gradient_finite(self, time, Lambda):
        h = Lambda * self.shape * time ** (self.shape - 1)
        S = tf.exp(-1 * Lambda * time ** self.shape)
        return tf.gradients(h * S, [time])[0]

    def gradient_infinite(self, time):
        return tf.zeros_like( time )



