import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from collections import Counter
from sklearn.utils import shuffle
from scipy.sparse import csr_matrix
from failure_rate_prediction_conf.data_entry_class.ImpressionEntry import MIN_OCCURRENCE_SYMBOL, MIN_OCCURRENCE


class SurvivalData:

    def __init__(self, times, events, sparse_features, sparse_headerbids, min_occurrence=MIN_OCCURRENCE):
        self.times, self.events, self.sparse_features, self.sparse_headerbids = \
            times, events, sparse_features.tocsr(), sparse_headerbids.tocsr()
        self.sparse_features_csc = sparse_features.tocsc()
        self.num_instances = len(self.times)
        self.max_nonzero_len = Counter(self.sparse_features.nonzero()[0]).most_common(1)[0][1]  # 94
        self.load_rares_index()

        self.infreq_user_col_indices, self.infreq_page_col_indices = np.array([]), np.array([])
        if min_occurrence > MIN_OCCURRENCE:
            self.infreq_user_col_indices, self.infreq_page_col_indices = self.load_addtl_infreq(min_occurrence)
            self.sparse_features = self.merge_addtl_infreq(self.sparse_features)


    def load_rares_index(self):
        attr2idx = pickle.load(open('output/attr2idx.dict', 'rb'))
        self.rare_user_col_index = attr2idx['UserId'][MIN_OCCURRENCE_SYMBOL]
        self.rare_page_col_index = attr2idx['NaturalIDs'][MIN_OCCURRENCE_SYMBOL]

    def load_addtl_infreq(self, min_occur):
        """
        Beside those users and pages whose occurrences are less than MIN_OCCURRENCE,
        we also merge those whose occurrences are less than min_occur (if min_occur > MIN_OCCURRENCE)

        :param min_occur:
        :return:
        """
        attr2idx = pickle.load(open('output/attr2idx.dict', 'rb'))
        counter = pickle.load(open('output/counter.dict', 'rb'))

        infreq_user_col_indices = np.array([attr2idx['UserId'][k]
                                           for k, v in counter['UserId'].items()
                                           if k in attr2idx['UserId'] and v < min_occur])
        infreq_page_col_indices = np.array([attr2idx['NaturalIDs'][k]
                                           for k, v in counter['NaturalIDs'].items()
                                           if k in attr2idx['NaturalIDs'] and v < min_occur])
        print("%d/%d users are additionally merged due to infrequency"
              % (len(infreq_user_col_indices), len(counter['UserId'])))
        print("%d/%d pages are additionally merged due to infrequency"
              % (len(infreq_page_col_indices), len(counter['NaturalIDs'])))
        return infreq_user_col_indices, infreq_page_col_indices

    def merge_addtl_infreq(self, sparse_features):
        if len(self.infreq_user_col_indices) > 0 or len(self.infreq_page_col_indices) > 0:
            infreq_user_row_mask = np.nonzero(sparse_features[:, self.infreq_user_col_indices])[0]
            infreq_user_row_indices = np.where(infreq_user_row_mask)[0]
            infreq_page_row_mask = np.nonzero(sparse_features[:, self.infreq_page_col_indices])[0]
            infreq_page_row_indices = np.where(infreq_page_row_mask)[0]

            # add 1 on the rare_user_index and rarw_page_index for the rows that have addtl infreq users or pages.
            sparse_features += self._get_rare_index_mask(sparse_features.shape,
                                                         infreq_user_row_indices,
                                                         self.rare_user_col_index)
            sparse_features += self._get_rare_index_mask(sparse_features.shape,
                                                         infreq_page_row_indices,
                                                         self.rare_page_col_index)

            # zero all addtl infreq users and pages
            # sparse_features -= self._get_infreq_index_mask(sparse_features.shape,
            #                                                infreq_user_row_indices,
            #                                                self.infreq_user_col_indices)
            # sparse_features -= self._get_infreq_index_mask(sparse_features.shape,
            #                                                infreq_page_row_indices,
            #                                                self.infreq_page_col_indices)
            return sparse_features

    def _get_rare_index_mask(self, shape, infreq_row_indices, rare_col_index):
        row = np.array(infreq_row_indices)
        col = np.array([rare_col_index] * len(infreq_row_indices))
        data = np.ones(len(infreq_row_indices))
        return csr_matrix((data, (row, col)), shape=shape, dtype=float)

    def _get_infreq_index_mask(self, shape, infreq_row_indices, infreq_col_indices):
        row = np.array(infreq_row_indices)
        col = np.array(infreq_col_indices)
        print(len(row), len(col))
        data = np.ones(len(infreq_row_indices))
        return csr_matrix((data, (row, col)), shape=shape, dtype=float)

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
            freq_user_mask = ~np.ravel(self.sparse_features[:, self.rare_user_col_index].toarray()).astype(bool)
            freq_page_mask = ~np.ravel(self.sparse_features[:, self.rare_page_col_index].toarray()).astype(bool)
            freq_both_mask = freq_user_mask & freq_page_mask
            self.times, self.events, self.sparse_features, self.sparse_headerbids = \
                self.times[freq_both_mask], self.events[freq_both_mask], \
                self.sparse_features[freq_both_mask], self.sparse_headerbids[freq_both_mask]
            assert self.times.shape[0] == self.events.shape[0] == self.sparse_features.shape[0] == self.sparse_headerbids.shape[0]

        infreq_col_set = set(self.infreq_user_col_indices) | set(self.infreq_page_col_indices)
        start_index = 0
        while start_index < self.num_instances:
            batch_feat_mat = self.sparse_features[start_index: start_index + batch_size, :]
            # batch_feat_mat = self.merge_addtl_infreq(batch_feat_mat)

            feat_indices_batch = np.split(batch_feat_mat.indices, batch_feat_mat.indptr)[1:-1]
            feat_values_batch = np.split(batch_feat_mat.data, batch_feat_mat.indptr)[1:-1]

            # feat_indices_batch = list(map(np.array,
            #                               (list(filter(lambda i: i not in infreq_col_set, indices_arr))
            #                                for indices_arr in feat_indices_batch)
            #                               )
            #                           )
            filter_mask = [np.array(list(map(lambda i: i not in infreq_col_set, indices_arr))).astype(bool) for indices_arr in feat_indices_batch]
            # print([len(f) for f in feat_indices_batch])
            # print([len(f) for f in feat_values_batch])
            feat_indices_batch = [feat_indices_batch[i][filter_mask[i]] for i in range(len(feat_indices_batch))]
            feat_values_batch = [feat_values_batch[i][filter_mask[i]] for i in range(len(feat_values_batch))]
            # print([len(f) for f in feat_indices_batch])
            # print([len(f) for f in feat_values_batch])


            # padding
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
    s = SurvivalData(times, events, sparse_features, sparse_headerbids, min_occurrence=10)

    for t, e, f_ind, f_val, h_ind, h_val, max_nonzero_len in s.make_sparse_batch(2048):
        print(t)
        print(e)
        print(f_ind)
        print(f_val)
        print(h_ind)
        print(h_val)
        print(max_nonzero_len)
        print()
