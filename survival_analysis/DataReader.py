import csv, pickle
from pprint import pprint
import tensorflow as tf
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle


class SurvivalData:

    def __init__(self, times, events, sparse_features):
        self.times, self.events, self.sparse_features = times, events, sparse_features.tocsr()
        self.num_instances = len(self.times)

    def make_batch(self, batch_size):
        shuffle(self.times, self.events, self.sparse_features)

        start_index = 0
        while start_index < self.num_instances:
            full_feat_mat = self.sparse_features.tocsr()[start_index: start_index + batch_size, :].toarray()
            full_feat_mat = np.zeros(shape=full_feat_mat.shape)
            yield self.times[start_index: start_index + batch_size], \
                  self.events[start_index: start_index + batch_size], \
                  full_feat_mat
            start_index += batch_size


if __name__ == "__main__":
    times, events, sparse_features = pickle.load(open('../Vectors_train.p', 'rb'))
    s = SurvivalData(times, events, sparse_features)
    for t, e, sf in s.make_batch(10):
        print(t)
        print(e)
        print(sf)


