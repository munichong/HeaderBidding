from pymongo import MongoClient
from survival_analysis.NetworkBackfillImpressionEntry import NetworkBackfillImpressionEntry

DBNAME = 'Header_Bidding'
COLNAME = 'NetworkBackfillImpressions'
FEATURE_FIELDS = ['URIs_pageno', 'NaturalIDs', 'RefererURL', 'UserId',
                  'DeviceCategory', 'MobileDevice', 'Browser', 'BandWidth', 'OS', 'MobileCarrier',
                  'SellerReservePrice', 'EstimatedBackfillRevenue',
                  'Time',
                  'RequestLanguage', 'Country', 'Region', 'Metro', 'City', 'AudienceSegmentIds',
                  'RequestedAdUnitSizes', 'CreativeSize', 'AdPosition',
                  'CustomTargeting', ]


class NetworkBackfillImpressionEntries:
    def __init__(self):
        client = MongoClient()
        self.col = client[DBNAME][COLNAME]

    def retrieve_entries(self):
        self.data = []
        for doc in self.col.find(projection=FEATURE_FIELDS):
            entry = NetworkBackfillImpressionEntry(doc)
            entry.build_entry()
            self.data.append(entry)


a = NetworkBackfillImpressionEntries()
a.retrieve_entries()





