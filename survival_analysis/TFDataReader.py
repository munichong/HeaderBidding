import tensorflow as tf
import numpy as np


class TFDataReader:

    def __init__(self, file_path, num_epochs, num_features):
        self.num_features = num_features
        self.dataset = tf.data.TFRecordDataset(file_path).repeat(num_epochs)

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
        row_cells = row.strip().split(",")
        duration, event, sparse_features = row_cells[0], row_cells[1], self._sparse2dense(row_cells[2:])
        return duration, event, sparse_features

    def _sparse2dense(self, sparse_vec):
        dense_vec = [0.0] * self.num_features
        for node in sparse_vec:
            index, value = node.split(':')
            dense_vec[int(index)] = float(value)
        return np.array(dense_vec)

