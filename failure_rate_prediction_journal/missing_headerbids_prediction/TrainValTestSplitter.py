"""
1. Given all impressions in MongoDB, filter the impressions whose at least one header bids are known.
2. Split them into training, validation, and test datasets.
"""
import os
import pickle
from random import shuffle

from failure_rate_prediction_journal.data_entry_class.ImpressionEntry import HEADER_BIDDING_KEYS
from failure_rate_prediction_journal.data_entry_class.ImpressionEntryGenerator import imp_entry_gen

BUFFER_QUEUE_SIZE = 80000

ROOT = '../output/missing_headerbids_data_partitions'

TRAIN_OUT_PATH, VAL_OUT_PATH, TEST_OUT_PATH = os.path.join(ROOT, 'TRAIN_SET'), \
                                              os.path.join(ROOT, 'VAL_SET'), \
                                              os.path.join(ROOT, 'TEST_SET')

TRAIN_PCT, VAL_PCT = 0.8, 0.1

''' Delete the files that were generated before '''
for file in os.listdir(ROOT):
    os.remove(os.path.join(ROOT, file))

''' For each header bidding agents, get all impressions in which their header bids are known '''
hb_known_impressions = [[] for _ in range(len(HEADER_BIDDING_KEYS))]
hb_known_file_index = [0] * len(HEADER_BIDDING_KEYS)
for imp_entry in imp_entry_gen():
    headerbids = imp_entry.get_headerbids()
    for i, hb in enumerate(headerbids):
        if hb is None:
            continue

        hb_known_impressions[i].append(imp_entry)

        if len(hb_known_impressions[i]) == BUFFER_QUEUE_SIZE:
            outfilename = HEADER_BIDDING_KEYS[i] + '_%d' % hb_known_file_index[i]

            ''' train, val, and test partitioning '''
            shuffle(hb_known_impressions[i])

            train_len = int(TRAIN_PCT * len(hb_known_impressions[i]))
            val_len = int(VAL_PCT * len(hb_known_impressions[i]))
            pickle.dump(hb_known_impressions[i][: train_len],
                        open(os.path.join(ROOT, outfilename + '_train.p'),
                             'wb'))
            pickle.dump(hb_known_impressions[i][train_len: train_len + val_len],
                        open(os.path.join(ROOT, outfilename + '_val.p'),
                             'wb'))
            pickle.dump(hb_known_impressions[i][train_len + val_len:],
                        open(os.path.join(ROOT, outfilename + '_test.p'),
                             'wb'))
            print('Files has been generated for %s.' % outfilename)
            hb_known_impressions[i].clear()
            hb_known_file_index[i] += 1
