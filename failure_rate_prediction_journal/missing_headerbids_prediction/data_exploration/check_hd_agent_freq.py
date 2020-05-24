import csv

import numpy as np
from pymongo import MongoClient

HEADER_BIDDING_KEYS = (
'mnetbidprice',  # Count = 4216971 (40.93%), Mean Value = 1.83 | Count = 4968367 (85.06%), Mean Value = 3.58
'mnet_abd',  # Count = 2902567 (28.17%), Mean Value = 1.69 | Count = 3692531 (63.22%), Mean Value = 3.25
'mnet_fbcpm',  # Count = 357871 (3.47%), Mean Value = 1.14 |　Count = 552611 (9.46%), Mean Value = 2.24
'amznbid',  # Count = 1395976 (13.55%), Mean Value = 0.83 | Count = 1803308 (30.87%), Mean Value = 2.07
'crt_pb',  # Count = 871658 (8.46%), Mean Value = 1.25 | Count = 1107635 (18.96%), Mean Value = 1.22
# 'fb_bid_price_cents', # no show

# 'amznslots',  # no show
# 'appnx_video_segment', # too few, large value, Count = 57860 (0.56%), Mean Value = 21.95 | Count = 150398 (2.57%), Mean Value = 25.80
# 'mnet_aat', # str, 'mnet_aat': 'o'
# 'mnet_asz' # not $, e.g., 'mnet_asz': '300x250'
# 'mnet_fb',  # no show
# 'mnet_fbsz', # not $, e.g., 'mnet_fbsz': '300x250'
# 'mnet_hv', # too few, Count = 13065 (0.13%), Mean Value = 1.00 | Count = 139510 (2.39%), Mean Value = 1.00
# 'mnet_placement', # not $, e.g., 'mnet_placement': 'rec-ad-article-0'
# 'mnet_video_segment', # too few, large value, Count = 40406 (0.39%), Mean Value = 34.08 | Count = 116438 (1.99%), Mean Value = 36.86
# 'mnetcid', # str, e.g. 'mnetcid': '8cu66o230'
# 'mnetdnb', # too few, mean is 1, Count = 2561262 (24.86%), Mean Value = 1.00 | Count = 122116 (2.09%), Mean Value = 1.00
# 'mnetsize', # not $, e.g., 'mnetsize': '320x50'
# 'tgrotslot'  # no show
)
AMZBID_MAPPING_PATH = '..\..\..\PricePoints-3038-display.csv'

client = MongoClient()
DBNAME = 'Header_Bidding'


def load_amznbid_price_mapping():
    amzbid_mapping = {}
    with open(AMZBID_MAPPING_PATH) as infile:
        csv_reader = csv.reader(infile, delimiter=',')
        next(csv_reader)
        for line in csv_reader:
            amzbid_mapping[line[-1]] = float(line[-2].replace('$', '').strip())
    return amzbid_mapping


amzbid_mapping = load_amznbid_price_mapping()


def get_headerbids_count(doc):
    ct = doc['CustomTargeting']
    count = [0] * len(HEADER_BIDDING_KEYS)
    for i, hb_key in enumerate(HEADER_BIDDING_KEYS):
        if hb_key.lower() not in ct:
            continue

        count[i] += 1
    return np.array(count)


def get_headerbids_value(doc):
    ct = doc['CustomTargeting']
    value = [0] * len(HEADER_BIDDING_KEYS)
    for i, hb_key in enumerate(HEADER_BIDDING_KEYS):
        if hb_key.lower() not in ct:
            continue

        if hb_key == 'amznbid':
            if ct[hb_key] in amzbid_mapping:
                value[i] = amzbid_mapping[ct[hb_key]]
        else:
            value[i] = float(ct[hb_key])
    return np.array(value)


for COLNAME in ['NetworkBackfillImpressions', 'NetworkImpressions']:
    col = client[DBNAME][COLNAME]
    total_entries = col.find().count()

    total_count = np.array([0] * len(HEADER_BIDDING_KEYS))
    values = np.array([0.0] * len(HEADER_BIDDING_KEYS))
    n = 0
    for doc in col.find(projection=['CustomTargeting'], no_cursor_timeout=True):
        if n % 1000000 == 0:
            print('%d/%d (%.2f%%)' % (n, total_entries, n / total_entries * 100))
        n += 1

        total_count += get_headerbids_count(doc)
        try:
            values += get_headerbids_value(doc)
        except Exception as e:
            print(e)
            print(doc['CustomTargeting'])
            exit(0)

    print(COLNAME, total_entries)
    for hb_agent, count, value in sorted(zip(HEADER_BIDDING_KEYS, total_count, values), key=lambda x: x[1],
                                         reverse=True):
        print('\t%s: Count = %d (%.2f%%), Mean Value = %.2f' % (
        hb_agent, count, count / total_entries * 100, value / (count + 1e-6)))
