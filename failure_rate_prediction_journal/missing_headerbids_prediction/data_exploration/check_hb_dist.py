import os
import matplotlib.pyplot as plt
from scipy import stats
from failure_rate_prediction_journal.data_entry_class.ImpressionEntry import HEADER_BIDDING_KEYS
from failure_rate_prediction_journal.missing_headerbids_prediction.DataReader import _load_headerbids_file


INPUT_DIR = '../../output'
ALL_AGENTS_DIR = os.path.join(INPUT_DIR, 'all_agents_vectorization')

hb_data = []
for i, hb_agent_name in enumerate(HEADER_BIDDING_KEYS):
        print("HB AGENT (%d/%d) %s" % (i + 1, len(HEADER_BIDDING_KEYS), hb_agent_name))
        for data_type in ['train', 'val', 'test']:
            hb_data.extend(_load_headerbids_file(ALL_AGENTS_DIR, hb_agent_name, data_type))


print()
print(stats.describe(hb_data))

sorted_hbs = sorted(hb_data)
for pct in [0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
    print('%d%% hbs >= %f' % (round((1 - pct) * 100), sorted_hbs[int(len(sorted_hbs) * pct)]))
