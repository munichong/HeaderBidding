import numpy as np
from pymongo import MongoClient
from failure_rate_prediction_journal.data_entry_class.ImpressionEntry import HEADER_BIDDING_KEYS
from failure_rate_prediction_journal.data_entry_class.ImpressionEntryGenerator import imp_entry_gen

client = MongoClient()

FEATURE_FIELDS = ['URIs_pageno', 'NaturalIDs', 'RefererURL', 'UserId',
                  'DeviceCategory', 'MobileDevice', 'Browser', 'BandWidth', 'OS', 'MobileCarrier',
                  'SellerReservePrice', 'EstimatedBackfillRevenue',
                  'Time',
                  'RequestLanguage', 'Country', 'Region', 'Metro', 'City',
                  'RequestedAdUnitSizes', 'AdPosition',
                  'CustomTargeting', ]

total_entires = 0
n = 0
counter = np.array([0] * len(HEADER_BIDDING_KEYS))
for imp_entry in imp_entry_gen():
    headerbids = imp_entry.get_headerbids()
    counter += np.array([int(hb!=None) for hb in headerbids])

print()
print("IN TOTAL: %d impressions: Pct: %s%%, Count: %s" %
    (total_entires, counter/total_entires*100.0, counter))