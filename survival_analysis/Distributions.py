import numpy as np
import tensorflow as tf

class WeibullDistribution:

    @staticmethod
    def gradient(time, Lambda, shape, threshold):
        '''

        :param time: i.e., lambda, tf.constant
        :param Lambda: regression
        :param shape: i.e., p, tf.constant
        :param threshold: tf.constant
        :return:
        '''
        if abs(time) == np.inf:
            return 0
        h = Lambda * shape * (time - threshold) ** (shape - 1)
        S = tf.exp(-1 * Lambda * (time - threshold) ** shape)
        return tf.gradients(h * S)



