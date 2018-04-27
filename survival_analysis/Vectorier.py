from pymongo import MongoClient
from pprint import pprint
from survival_analysis.ImpressionEntry import ImpressionEntry
from collections import defaultdict, Counter


FEATURE_FIELDS = ['URIs_pageno', 'NaturalIDs', 'RefererURL', 'UserId',
                  'DeviceCategory', 'MobileDevice', 'Browser', 'BandWidth', 'OS', 'MobileCarrier',
                  'SellerReservePrice', 'EstimatedBackfillRevenue',
                  'Time',
                  'RequestLanguage', 'Country', 'Region', 'Metro', 'City',
                  'RequestedAdUnitSizes', 'AdPosition',
                  'CustomTargeting', ]


class Vectorier:
    def __init__(self, dbname, colname):
        self.dbname = dbname
        self.colname = colname
        client = MongoClient()
        self.col = client[dbname][colname]
        self.counter = defaultdict(Counter)  # {Attribute1:set(feat1, feat2, ...), Attribute2: set(feat1, ...), ...}

    def count_unique_attributes(self):
        n = 0
        total_entires = self.col.find().count()
        for doc in self.col.find(projection=FEATURE_FIELDS):
            if n % 1000000 == 0:
                print('%d/%d' % (n, total_entires))
                # print(self.table)
            n += 1

            imp_entry = ImpressionEntry(doc)
            imp_entry.build_entry()
            for k, v in imp_entry.entry.items():
                if type(v) == list:
                    self.counter[k].update(v)
                else:
                    self.counter[k][v] += 1

    def build_attr2idx(self):
        self.attr2idx = defaultdict(dict)
        for attr, feat_counter in self.counter.items():
            current_index = 0  # reset for every attribute
            for feat in self.counter[attr]:
                self.attr2idx[attr][feat] = current_index
                current_index += 1


if __name__ == "__main__":
    vectorizer = Vectorier('Header_Bidding', 'NetworkBackfillImpressions')
    vectorizer.count_unique_attributes()
    vectorizer.build_attr2idx()
    pprint(vectorizer.counter)
    pprint(vectorizer.attr2idx)

