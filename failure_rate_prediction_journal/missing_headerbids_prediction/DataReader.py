import os
import numpy as np
from scipy import sparse
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
from sklearn.utils import shuffle

class HeaderBiddingData:

    def __init__(self):
        self.headerbids = []
        self.sparse_features = None
        self.max_nonzero_len = 0

    def num_instances(self):
        return self.sparse_features.shape[0]

    def num_features(self):
        return self.sparse_features.shape[1]

    def add_data(self, headerbids : list,
                 sparse_features : sparse.csr.csr_matrix):
        self.headerbids.extend(headerbids)

        if self.sparse_features is None:
            self.sparse_features = sparse_features
        else:
            self.sparse_features = sparse.vstack(
                (self.sparse_features,
                 sparse_features)
            )

        assert self.sparse_features.shape[0] == len(self.headerbids)

        self.max_nonzero_len = max(self.max_nonzero_len,
                                   Counter(self.sparse_features.nonzero()[0]).most_common(1)[0][1]
                                   )

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


def load_hb_data(hb_agent_name):
    sparse_features = sparse.load_npz(os.path.join(INPUT_DIR,
                                                   '%s_featvec_train.csr.npz' % hb_agent_name))
    headerbids = list(
        map(
            float,
            open(os.path.join(INPUT_DIR,
                              '%s_headerbids_train.csv' % hb_agent_name))
                .read().splitlines()
        )
    )
    return headerbids, sparse_features

if __name__ == "__main__":
    INPUT_DIR = '../output/all_agents_vectorization'

    s = HeaderBiddingData()

    for i, hd_agent_name in enumerate(['mnetbidprice', 'amznbid']):
        headerbids, sparse_features = load_hb_data(hd_agent_name)
        print("%d instances and %d features" % sparse_features.shape)

        hb_agent_onehot = [0.0] * 2
        hb_agent_onehot[i] = 1.0
        sparse_features = sparse.hstack(
            (np.array([hb_agent_onehot]*sparse_features.shape[0]),
            sparse_features)
        )
        print("\t%d instances and %d features" % sparse_features.shape)

        s.add_data(headerbids, sparse_features)

    for hb, f_ind, f_val, max_nonzero_len in s.make_sparse_batch(1):
        print(hb)
        print(f_ind)
        print(f_val)
        print(max_nonzero_len)
        print()
