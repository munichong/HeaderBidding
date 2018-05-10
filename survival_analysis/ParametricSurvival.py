import numpy as np
import tensorflow as tf

class ParametricSurvivalModel:

    def __init__(self):
        pass

    def run_graph(self, distribution, num_features, k=1):
        '''

        :param distribution:
        :param num_features:
        :param k: the dimensionality of the embedding, Must be >= 1
        :return:
        '''
        # INPUTs
        is_training = tf.placeholder(tf.bool, shape=(), name='bool_train')
        input_vectors = tf.placeholder(tf.float32,
                                  shape=[None, num_features],
                                  name='input_vectors')

        embedding = tf.Variable(tf.truncated_normal(shape=(num_features, k), mean=0.0, stddev=0.5))




        if k == 1:
            ''' The hazard is a linear regression '''

            ''' treat the input_vectors as masks '''
            ''' input_vectors do NOT need to be binary vectors '''
            feat_vals = tf.expand_dims(input_vectors, axis=-1)
            feat_vals = tf.tile(feat_vals, [1, k])
            feat_x_weights = tf.reduce_sum(tf.multiply(embedding, feat_vals), axis=1)

            intercept = tf.Variable(tf.constant(0.1))
            Lambda = feat_x_weights + intercept
        else:
            pass






