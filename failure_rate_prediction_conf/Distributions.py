import tensorflow as tf


class WeibullDistribution:
    def __init__(self, threshold=0.05):
        '''

        :param shape: The Weibull distribution with shape = 1 is the exponential distribution
                      A value of shape > 1 indicates that the failure rate increases over time.
        :param threshold:
        '''
        self.threshold = threshold

    def left_censoring(self, time, scale, shape=0.2):
        return 1 - self.right_censoring(time, scale, shape)

    def right_censoring(self, time, scale, shape):
        # with time increases, the return value should decrease
        return tf.exp(-1 * scale * time ** shape)


class LogLogisticDistribution:
    def __init__(self):
        pass

    def left_censoring(self, time, scale, shape=0.7):
        return 1 - self.right_censoring(time, scale, shape)

    def right_censoring(self, time, scale, shape):
        return 1 / (scale * time ** shape + 1)


#     def left_censoring(self, time, scale):
#         return 1 / (time / scale) ** (-1 * self.shape)
# #
#     def right_censoring(self, time, scale):
#         return 1 - self.left_censoring(time, scale)

class LogisticDistribution:
    def __init__(self):
        pass

    def cdf_pointwise(self, time, scale, shape):
        return 1 / (1 + tf.exp((shape - time) / scale))

    def left_censoring(self, time, scale, shape=0.4):
        return self.cdf_pointwise(time, scale, shape) - self.cdf_pointwise(0.0, scale, shape)

    def right_censoring(self, time, scale, shape):
        return 1 - self.cdf_pointwise(time, scale, shape)


class GammaDistribution:
    def __init__(self):
        pass

    def left_censoring(self, time, scale, shape=1.0):
        gamma_dist = tf.distributions.Gamma(concentration=shape, rate=scale)
        return gamma_dist.cdf(time)

    def right_censoring(self, time, scale, shape=1.0):
        return 1 - self.left_censoring(time, scale, shape)


class GumbelDistribution:
    def __init__(self):
        pass

    def double_exp_part(self, time, scale, shape):  # shape is actually called "location" in Statistics
        return tf.exp(-1 * tf.exp((shape - time) / scale))

    def left_censoring(self, time, scale, shape=0.001):
        return self.double_exp_part(time, scale, shape) - self.double_exp_part(0.0, scale, shape)

    def right_censoring(self, time, scale, shape):
        return 1 - self.double_exp_part(time, scale, shape)


class ExponentialDistribution:
    def __init__(self):
        pass

    def left_censoring(self, time, scale, shape):
        return 1 - self.right_censoring(time, scale)

    def right_censoring(self, time, scale):
        return tf.exp(-1 * scale * time)
