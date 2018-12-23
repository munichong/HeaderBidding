import numpy as np
from pymongo import MongoClient
from failure_rate_prediction_journal.data_entry_class.ImpressionEntry import HEADER_BIDDING_KEYS
from failure_rate_prediction_journal.data_entry_class.NetworkBackfillImpressionEntry import NetworkBackfillImpressionEntry
from failure_rate_prediction_journal.data_entry_class.NetworkImpressionEntry import NetworkImpressionEntry

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
DBNAME = 'Header_Bidding'
for COLNAME, ImpressionEntry in [('NetworkBackfillImpressions', NetworkBackfillImpressionEntry),
                                 ('NetworkImpressions', NetworkImpressionEntry)]:
    col = client[DBNAME][COLNAME]
    total_entires += col.find().count()
    for doc in col.find(projection=FEATURE_FIELDS):
        if n % 1000000 == 0:
            print('%d/%d' % (n, total_entires))
            if n > 0:
                print("SO FAR: %d impressions: Pct(%%): %s, Count: %s" %
                      (n, counter / n * 100.0, counter))
        n += 1

        imp_entry = ImpressionEntry(doc)
        imp_entry.build_entry()

        if not imp_entry.is_qualified():
            continue

        headerbids = imp_entry.get_headerbids()
        counter += np.array([int(hb!=None) for hb in headerbids])

print()
print("IN TOTAL: %d impressions: Pct: %s, Count: %s" %
    (total_entires, counter/total_entires*100.0, counter))