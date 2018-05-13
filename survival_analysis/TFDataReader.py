import csv
import tensorflow as tf
import numpy as np


class TFDataReader:

    def __init__(self, file_path, num_epochs, num_features):
        self.num_features = num_features
        # self.dataset = tf.data.TFRecordDataset(file_path).repeat(num_epochs)
        self.dataset = tf.data.TextLineDataset(file_path).repeat(num_epochs)
        self.num_data = sum(1 for _ in csv.reader(open(file_path))) - 1  # minus the header line

    def make_batch(self, batch_size):
        # Parse records
        self.dataset = self.dataset.map(self._inputstr2vector, num_parallel_calls=2)

        # Shuffle the dataset
        self.dataset = self.dataset.shuffle(buffer_size=batch_size * 10,
                                            reshuffle_each_iteration=True)

        # Batch it up.
        self.dataset = self.dataset.batch(batch_size)
        iterator = self.dataset.make_one_shot_iterator()
        duration_batch, event_batch, features_batch = iterator.get_next()

        return duration_batch, event_batch, features_batch

    def _inputstr2vector(self, row):
        row_cells = tf.string_split([row], ',', skip_empty=True).values

        duration, event, sparse_features = row_cells[0], row_cells[1], tf.map_fn(self._sparsevec2densevec, row_cells[2:])
        return duration, event, sparse_features

    def _sparsevec2densevec(self, row):

        sparse_indices = tf.map_fn(self._sparsenode2densenode, [row])

        output_shape = tf.constant([self.num_features], dtype=tf.int32)
        sparse_values = tf.constant([1.0] * sparse_indices.shape[0], dtype=tf.float32)
        return tf.sparse_to_dense(sparse_indices, output_shape, sparse_values)

    def _sparsenode2densenode(self, sparse_node):
        index, value = tf.string_split([sparse_node], ':').values

        return int(index)

