import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from collections import Counter
from sklearn.utils import shuffle
from failure_rate_prediction_conf.data_entry_class.ImpressionEntry import MIN_OCCURRENCE_SYMBOL


class SurvivalData:

    def __init__(self, times, events, sparse_features, sparse_headerbids):
        self.times, self.events, self.sparse_features, self.sparse_headerbids = \
            times, events, sparse_features.tocsr(), sparse_headerbids.tocsr()
        self.num_instances = len(self.times)
        self.max_nonzero_len = Counter(self.sparse_features.nonzero()[0]).most_common(1)[0][1]  # 94
        self.load_rares_index()

    def load_rares_index(self):
        attr2idx = pickle.load(open('output/attr2idx.dict', 'rb'))
        self.rare_user_index = attr2idx['UserId'][MIN_OCCURRENCE_SYMBOL]
        self.rare_page_index = attr2idx['NaturalIDs'][MIN_OCCURRENCE_SYMBOL]

    def get_sparse_feat_vec_batch(self, batch_size=100):
        '''
        For baselines
        :param batch_size:
        :return:
        '''
        self.times, self.events, self.sparse_features, self.sparse_headerbids = \
            shuffle(self.times, self.events, self.sparse_features, self.sparse_headerbids)

        start_index = 0
        while start_index < self.num_instances:

            yield self.times[start_index: start_index + batch_size], \
                  self.events[start_index: start_index + batch_size], \
                  self.sparse_features[start_index: start_index + batch_size, :]
            start_index += batch_size

    def make_sparse_batch(self, batch_size=10000, only_freq=False):
        '''
        for our methods
        :param batch_size:
        :return:
        '''
        self.times, self.events, self.sparse_features, self.sparse_headerbids = \
            shuffle(self.times, self.events, self.sparse_features, self.sparse_headerbids)
        '''
        self.times: <class 'numpy.ndarray'>
        self.events: <class 'numpy.ndarray'>
        self.sparse_features: <class 'scipy.sparse.csr.csr_matrix'>
        self.sparse_headerbids: <class 'scipy.sparse.csr.csr_matrix'>
        '''

        if only_freq:
            freq_user_mask = ~np.ravel(self.sparse_features[:, self.rare_user_index].toarray()).astype(bool)
            freq_page_mask = ~np.ravel(self.sparse_features[:, self.rare_page_index].toarray()).astype(bool)
            freq_both_mask = freq_user_mask & freq_page_mask
            self.times, self.events, self.sparse_features, self.sparse_headerbids = \
                self.times[freq_both_mask], self.events[freq_both_mask], \
                self.sparse_features[freq_both_mask], self.sparse_headerbids[freq_both_mask]
            assert self.times.shape[0] == self.events.shape[0] == self.sparse_features.shape[0] == self.sparse_headerbids.shape[0]


        start_index = 0
        while start_index < self.num_instances:
            batch_feat_mat = self.sparse_features[start_index: start_index + batch_size, :]
            # padding
            feat_indices_batch = np.split(batch_feat_mat.indices, batch_feat_mat.indptr)[1:-1]
            feat_values_batch = np.split(batch_feat_mat.data, batch_feat_mat.indptr)[1:-1]
            feat_indices_batch = pad_sequences(feat_indices_batch, maxlen=self.max_nonzero_len, padding='post', value=0)
            feat_values_batch = pad_sequences(feat_values_batch, maxlen=self.max_nonzero_len, padding='post', value=0.0, dtype='float32')

            batch_hb_mat = self.sparse_headerbids[start_index: start_index + batch_size, :]
            '''
            EXAMPLE of row (Array):
            [0.1]
            [2.77]
            []
            [0.04 1.52 1.53]
            [1.56 1.5 ]
            []
            [0.12 0.03 0.22]
            '''
            min_hbs_batch = []
            max_hbs_batch = []
            for row in np.split(batch_hb_mat.data, batch_hb_mat.indptr)[1:-1]:
                if row.size:
                    min_hbs_batch.append(min(row))
                    max_hbs_batch.append(max(row))
                else:
                    # if header bids are missing, use 0.0 instead.
                    min_hbs_batch.append(0.0)
                    max_hbs_batch.append(0.0)


            yield self.times[start_index: start_index + batch_size], \
                  self.events[start_index: start_index + batch_size], \
                  feat_indices_batch, \
                  feat_values_batch, \
                  min_hbs_batch, \
                  max_hbs_batch, \
                  self.max_nonzero_len
            start_index += batch_size



if __name__ == "__main__":
    times, events, sparse_features, sparse_headerbids = pickle.load(open('output/TRAIN_SET.p', 'rb'))
    s = SurvivalData(times, events, sparse_features, sparse_headerbids)

    for t, e, f_ind, f_val, h_ind, h_val, max_nonzero_len in s.make_sparse_batch(100):
        print(t)
        print(e)
        print(f_ind)
        print(f_val)
        print(h_ind)
        print(h_val)
        print(max_nonzero_len)
        print()
