import tensorflow as tf
import numpy as np
from scipy.sparse import csr_matrix

# sparse_indices = tf.constant([2, 5], dtype=tf.int32)
# output_shape = tf.constant([10], dtype=tf.int32)
# sparse_values = tf.constant([1] * sparse_indices.shape[0], dtype=tf.float32)
#
# a = tf.sparse_to_dense(sparse_indices, output_shape, sparse_values)
# b = tf.string_split(["123:1", "234:1"], ':')
#
# def _sparsenode2densenode(cell):
#     return tf.string_split(cell, ':').values
#
# def _sparsevec2densevec(row):
#     sparse_indices = tf.map_fn(_sparsenode2densenode, tf.convert_to_tensor(row))
#     return tf.convert_to_tensor(row)
#
#
# row_cells = tf.string_split(["123:1,234:1"], ',', skip_empty=True).values
# a = tf.map_fn(_sparsevec2densevec, row_cells[:])
#
#
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     init.run()
#     print(sess.run(a))

a = csr_matrix(np.arange(12).reshape((4,3)))
print(a)