import os
import csv
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from scipy.sparse import coo_matrix
from collections import Counter
from sklearn.utils import shuffle

class HeaderBidingData:

    def __init__(self, headerbids_file, sparse_features_file):

        self.headerbids = np.array([float(hb.rstrip()) for hb in open(headerbids_file).readlines()])
        self.sparse_features = self.featstr_to_sparsemat(sparse_features_file)
        self.num_instances = len(self.headerbids)
        self.max_nonzero_len = Counter(self.sparse_features.nonzero()[0]).most_common(1)[0][1]

    def featstr_to_sparsemat(self, file_path):
        row_indices_fv, col_indices_fv, values_fv = [], [], []
        num_rows = 0
        row_gen = csv.reader(open(file_path))
        num_features = int(next(row_gen)[0])
        for row in row_gen:
            for node in row:
                col_index, val = node.split(':')
                row_indices_fv.append(num_rows)
                col_indices_fv.append(int(col_index))
                values_fv.append(float(val))
            num_rows += 1
        return coo_matrix((values_fv, (row_indices_fv, col_indices_fv)), shape=(num_rows, num_features)).tocsr()

    # def get_sparse_feat_vec_batch(self, batch_size=100):
    #     #     self.headerbids, self.sparse_features = shuffle(self.headerbids, self.sparse_features)
    #     #
    #     #     start_index = 0
    #     #     while start_index < self.num_instances:
    #     #
    #     #         yield self.headerbids[start_index: start_index + batch_size], \
    #     #               self.sparse_features[start_index: start_index + batch_size, :]
    #     #         start_index += batch_size

    def make_sparse_batch(self, batch_size=10000):
        # self.headerbids, self.sparse_features = shuffle(self.headerbids, self.sparse_features)

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
    sparse_features_file = os.path.join(INPUT_DIR, 'mnetbidprice_featvec_train.csv')
    headerbids_file = os.path.join(INPUT_DIR, 'mnetbidprice_headerbids_train.csv')
    s = HeaderBidingData(headerbids_file, sparse_features_file)

    for hb, f_ind, f_val, max_nonzero_len in s.make_sparse_batch(10):
        print(hb)
        print(f_ind)
        print(f_val)
        print(max_nonzero_len)
        print()
