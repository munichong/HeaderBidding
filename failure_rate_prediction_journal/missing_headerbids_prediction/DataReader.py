import os
import numpy as np
from scipy import sparse
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
from sklearn.utils import shuffle
from failure_rate_prediction_conf.data_entry_class.ImpressionEntry import HEADER_BIDDING_KEYS


HB_OUTLIER_THLD = 5.0

class HeaderBiddingData:

    def __init__(self):
        self.headerbids = []
        self.sparse_features = None
        self.max_nonzero_len = 0

    def num_instances(self):
        return self.sparse_features.shape[0]

    def num_features(self):
        return self.sparse_features.shape[1]

    def add_data(self, headerbids : np.array,
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
        while start_index < self.num_instances():
            batch_feat_mat = self.sparse_features[start_index: start_index + batch_size, :]
            # print(batch_feat_mat)
            # padding
            feat_indices_batch = np.split(batch_feat_mat.indices, batch_feat_mat.indptr)[1:-1]
            feat_values_batch = np.split(batch_feat_mat.data, batch_feat_mat.indptr)[1:-1]
            feat_indices_batch = pad_sequences(feat_indices_batch, maxlen=self.max_nonzero_len,
                                               padding='post', value=0)
            feat_values_batch = pad_sequences(feat_values_batch, maxlen=self.max_nonzero_len,
                                              padding='post', value=0.0, dtype='float32')
            yield self.headerbids[start_index: start_index + batch_size], \
                  feat_indices_batch, \
                  feat_values_batch
            start_index += batch_size


def _load_sparsefeatures_file(dir_path, hb_agent_name, data_type):
    return sparse.load_npz(os.path.join(dir_path,
                                        '%s_featvec_%s.csr.npz' %
                                        (hb_agent_name, data_type)))

def _load_headerbids_file(dir_path, hb_agent_name, data_type):
    return list(
        map(
            float,
            open(os.path.join(dir_path,
                              '%s_headerbids_%s.csv' %
                              (hb_agent_name, data_type)))
                .read().splitlines()
        )
    )

def _filter_outliers(headerbids, sparse_features):
    headerbids = np.array(headerbids)
    mask = headerbids < HB_OUTLIER_THLD
    assert sparse_features.shape[0] == len(headerbids) == len(mask)
    headerbids = headerbids[mask]
    sparse_features = sparse_features[mask, :]
    assert sparse_features.shape[0] == len(headerbids)
    return headerbids, sparse_features

def load_hb_data_one_agent(dir_path, hb_agent_name, data_type):
    sparse_features = _load_sparsefeatures_file(dir_path, hb_agent_name, data_type)
    headerbids = _load_headerbids_file(dir_path, hb_agent_name, data_type)

    assert sparse_features.shape[0] == len(headerbids)

    print("\tORIGINAL: %d *%s* instances and %d features" % (sparse_features.shape[0],
                                                                        data_type,
                                                                        sparse_features.shape[1]))
    headerbids, sparse_features = _filter_outliers(headerbids, sparse_features)
    print("\tAFTER FILTERING OUTLIERS: %d *%s* instances and %d features" % (sparse_features.shape[0],
                                                                        data_type,
                                                                        sparse_features.shape[1]))
    return headerbids, sparse_features

def load_hb_data_all_agents(dir_path, hb_agent_name, data_type):
    headerbids, sparse_features = load_hb_data_one_agent(dir_path, hb_agent_name, data_type)

    # add hb_agent as one additional feature
    hb_agent_onehot = [float(hb_agent_name == agent) for agent in HEADER_BIDDING_KEYS]
    hb_agent_onehot = hb_agent_onehot[:-1]  # skip the last one for dummy variable
    sparse_features = sparse.hstack(
        (np.array([hb_agent_onehot] * sparse_features.shape[0]),
         sparse_features)
    )
    print("\tAFTER ADDING AGENT: %d *%s* instances and %d features" % (sparse_features.shape[0],
                                                                      data_type,
                                                                      sparse_features.shape[1]))

    return headerbids, sparse_features



if __name__ == "__main__":
    INPUT_DIR = '../output/all_agents_vectorization'

    train_data = HeaderBiddingData()

    for i, hd_agent_name in enumerate(['mnetbidprice', 'amznbid']):
        headerbids, sparse_features = load_hb_data_all_agents(INPUT_DIR, hd_agent_name, 'train')
        train_data.add_data(headerbids, sparse_features)

    for hb, f_ind, f_val in train_data.make_sparse_batch(1):
        print(hb)
        print(f_ind)
        print(f_val)
        print()
