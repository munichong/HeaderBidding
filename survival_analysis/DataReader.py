import csv, pickle
from pprint import pprint
import tensorflow as tf
import numpy as np
from collections import Counter
from sklearn.utils import shuffle


class SurvivalData:

    def __init__(self, times, events, sparse_features):
        self.times, self.events, self.sparse_features = times, events, sparse_features.tocsr()
        self.num_instances = len(self.times)

    def make_dense_batch(self, batch_size):
        self.times, self.events, self.sparse_features = shuffle(self.times, self.events, self.sparse_features)

        start_index = 0
        while start_index < self.num_instances:
            batch_feat_mat = self.sparse_features[start_index: start_index + batch_size, :].toarray()
            # full_feat_mat = np.zeros(shape=full_feat_mat.shape)
            yield self.times[start_index: start_index + batch_size], \
                  self.events[start_index: start_index + batch_size], \
                  batch_feat_mat
            start_index += batch_size

    def make_sparse_batch(self, batch_size):
        self.times, self.events, self.sparse_features = shuffle(self.times, self.events, self.sparse_features)
        max_nonzero_len = Counter(self.sparse_features.nonzero()[0]).most_common(1)[0][1]
        # print(max_nonzero_len)  # 103

        start_index = 0
        while start_index < self.num_instances:
            batch_feat_mat = self.sparse_features[start_index: start_index + batch_size, :]
            feat_indices_batch = [list(row) + [0.0] * (max_nonzero_len - len(row))
                                  for row in np.split(batch_feat_mat.indices, batch_feat_mat.indptr)[1:-1]]
            # print(feat_indices_batch)
            feat_values_batch = [list(row) + [0.0] * (max_nonzero_len - len(row))
                                  for row in np.split(batch_feat_mat.data, batch_feat_mat.indptr)[1:-1]]

            yield self.times[start_index: start_index + batch_size], \
                  self.events[start_index: start_index + batch_size], \
                  feat_indices_batch, \
                  feat_values_batch
            start_index += batch_size



if __name__ == "__main__":
    times, events, sparse_features = pickle.load(open('../Vectors_train.p', 'rb'))
    s = SurvivalData(times, events, sparse_features)
    for t, e, ind, val in s.make_sparse_batch(10):
        print(t)
        print(e)
        print(ind)
        print(val)
        print()



