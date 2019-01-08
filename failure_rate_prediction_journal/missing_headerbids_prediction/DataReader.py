import os
import numpy as np
from scipy import sparse
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
from sklearn.utils import shuffle

class HeaderBidingData:

    def __init__(self, headerbids : list,
                 sparse_features : sparse.csr.csr_matrix):

        self.headerbids = headerbids
        self.sparse_features = sparse_features
        self.num_instances = len(self.headerbids)
        assert self.sparse_features.shape[0] == self.num_instances

        self.max_nonzero_len = Counter(self.sparse_features.nonzero()[0]).most_common(1)[0][1]

    def make_sparse_batch(self, batch_size=10000):
        self.headerbids, self.sparse_features = shuffle(self.headerbids, self.sparse_features)

        start_index = 0
        while start_index < self.num_instances:
            batch_feat_mat = self.sparse_features[start_index: start_index + batch_size, :]
            # padding
            feat_indices_batch = np.split(batch_feat_mat.indices, batch_feat_mat.indptr)[1:-1]
            feat_values_batch = np.split(batch_feat_mat.data, batch_feat_mat.indptr)[1:-1]
            feat_indices_batch = pad_sequences(feat_indices_batch, maxlen=self.max_nonzero_len, padding='post', value=0)
            feat_values_batch = pad_sequences(feat_values_batch, maxlen=self.max_nonzero_len, padding='post', value=0.0, dtype='float32')
            yield self.headerbids[start_index: start_index + batch_size], \
                  feat_indices_batch, \
                  feat_values_batch, \
                  self.max_nonzero_len
            start_index += batch_size



if __name__ == "__main__":
    INPUT_DIR = '../output/all_agents_vectorization'
    sparse_features = sparse.load_npz(os.path.join(INPUT_DIR, 'mnetbidprice_featvec_train.csr.npz'))
    headerbids = list(
        map(
            float,
            open(os.path.join(INPUT_DIR,
                              'mnetbidprice_headerbids_train.csv'))
                .read().splitlines()
        )
    )
    s = HeaderBidingData(headerbids, sparse_features)

    for hb, f_ind, f_val, max_nonzero_len in s.make_sparse_batch(1):
        print(hb)
        print(f_ind)
        print(f_val)
        print(max_nonzero_len)
        print()
