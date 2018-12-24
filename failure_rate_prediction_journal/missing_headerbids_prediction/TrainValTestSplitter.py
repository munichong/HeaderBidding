import pickle
from random import shuffle
from failure_rate_prediction_journal.data_entry_class.ImpressionEntryGenerator import imp_entry_gen
from failure_rate_prediction_journal.data_entry_class.ImpressionEntry import HEADER_BIDDING_KEYS


TRAIN_OUT_PATH, VAL_OUT_PATH, TEST_OUT_PATH = '../output/TRAIN_SET', '../output/VAL_SET', '../output/TEST_SET'

TRAIN_PCT, VAL_PCT = 0.8, 0.1

''' For each header bidding agents, get all impressions in which their header bids are known '''
hb_known_impressions = [[] for _ in range(len(HEADER_BIDDING_KEYS))]
for imp_entry in imp_entry_gen():
    headerbids = imp_entry.get_headerbids()
    for i, hb in enumerate(headerbids):
        if hb is not None:
            hb_known_impressions[i].append(imp_entry)


''' Shuffle, split, and store '''
for i, agent_impressions in enumerate(hb_known_impressions):
    agent_name = HEADER_BIDDING_KEYS[i]
    print("AGENT_NAME: %s" % agent_name)
    n_train, n_val = int(len(agent_impressions) * TRAIN_PCT), int(len(agent_impressions) * VAL_PCT)

    shuffle(agent_impressions)
    imp_train = agent_impressions[ : n_train]
    imp_val = agent_impressions[n_train : n_train + n_val]
    imp_test = agent_impressions[n_train + n_val : ]

    train_file_path = '_'.join([TRAIN_OUT_PATH, agent_name, 'train.p'])
    val_file_path = '_'.join([VAL_OUT_PATH, agent_name, 'val.p'])
    test_file_path = '_'.join([TEST_OUT_PATH, agent_name, 'test.p'])
    pickle.dump(imp_train, open(train_file_path, 'wb'))
    print("Dumped %d training impressions in %s" % (len(imp_train), train_file_path))
    pickle.dump(imp_val, open(val_file_path, 'wb'))
    print("Dumped %d validation impressions in %s" % (len(imp_val), val_file_path))
    pickle.dump(imp_test, open(test_file_path, 'wb'))
    print("Dumped %d test impressions in %s" % (len(imp_test), test_file_path))
