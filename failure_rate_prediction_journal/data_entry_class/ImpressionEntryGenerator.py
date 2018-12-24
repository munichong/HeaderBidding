from pymongo import MongoClient
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

def imp_entry_gen():
    n = 0
    DBNAME = 'Header_Bidding'
    for COLNAME, ImpressionEntry in [('NetworkBackfillImpressions', NetworkBackfillImpressionEntry),
                                     ('NetworkImpressions', NetworkImpressionEntry)]:
        col = client[DBNAME][COLNAME]
        total_entires = col.find().count()
        for doc in col.find(projection=FEATURE_FIELDS):
            if n % 1000000 == 0:
                print('%d/%d' % (n, total_entires))
            n += 1

            imp_entry = ImpressionEntry(doc)
            imp_entry.build_entry()

            if not imp_entry.is_qualified():
                continue

            yield imp_entry

