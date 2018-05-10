import numpy as np
import tensorflow as tf

class WeibullDistribution:

    def __init__(self, shape=1, threshold=0.05):
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
        if abs(time) == np.inf:
            return 0
        h = Lambda * self.shape * time ** (self.shape - 1)
        S = tf.exp(-1 * Lambda * time ** self.shape)
        return tf.gradients(h * S)



