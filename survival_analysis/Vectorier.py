from pymongo import MongoClient
from survival_analysis.ImpressionEntry import ImpressionEntry
from collections import defaultdict

DBNAME = 'Header_Bidding'
COLNAME = 'NetworkBackfillImpressions'
FEATURE_FIELDS = ['URIs_pageno', 'NaturalIDs', 'RefererURL', 'UserId',
                  'DeviceCategory', 'MobileDevice', 'Browser', 'BandWidth', 'OS', 'MobileCarrier',
                  'SellerReservePrice', 'EstimatedBackfillRevenue',
                  'Time',
                  'RequestLanguage', 'Country', 'Region', 'Metro', 'City', 'AudienceSegmentIds',
                  'RequestedAdUnitSizes', 'CreativeSize', 'AdPosition',
                  'CustomTargeting', ]


class Vectorier:
    def __init__(self):
        client = MongoClient()
        self.col = client[DBNAME][COLNAME]
        self.table = defaultdict(set)  # {Attribute1:set(feat1, feat2, ...), Attribute2: set(feat1, ...), ...}

    def retrieve_entries(self):
        for doc in self.col.find(projection=FEATURE_FIELDS):
            imp_entry = ImpressionEntry(doc)
            imp_entry.build_entry()
            for k, v in imp_entry.entry.items():
                if type(v) == list:
                    self.table[k] = self.table[k].union(v)
                else:
                    self.table[k].add(v)


vectorizer = Vectorier()
vectorizer.retrieve_entries()

