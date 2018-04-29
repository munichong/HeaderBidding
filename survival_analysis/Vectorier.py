import numpy as np
from pymongo import MongoClient
from pprint import pprint
from survival_analysis.NetworkBackfillImpressionEntry import NetworkBackfillImpressionEntry
from survival_analysis.NetworkImpressionEntry import NetworkImpressionEntry
from collections import defaultdict, Counter


FEATURE_FIELDS = ['URIs_pageno', 'NaturalIDs', 'RefererURL', 'UserId',
                  'DeviceCategory', 'MobileDevice', 'Browser', 'BandWidth', 'OS', 'MobileCarrier',
                  'SellerReservePrice', 'EstimatedBackfillRevenue',
                  'Time',
                  'RequestLanguage', 'Country', 'Region', 'Metro', 'City',
                  'RequestedAdUnitSizes', 'AdPosition',
                  'CustomTargeting', ]


class Vectorier:
    def __init__(self):
        self.client = MongoClient()
        self.counter = defaultdict(Counter)  # {Attribute1:set(feat1, feat2, ...), Attribute2: set(feat1, ...), ...}

    def fit(self, dbname, colname, ImpressionEntry):
        '''count unique attributes'''
        self.col = self.client[dbname][colname]

        STOP = False
        n = 0
        total_entires = self.col.find().count()
        for doc in self.col.find(projection=FEATURE_FIELDS):
            if n % 1000000 == 0:
                print('%d/%d' % (n, total_entires))
                if STOP:
                    break
                STOP = True
            n += 1

            imp_entry = ImpressionEntry(doc)
            imp_entry.build_entry()
            for k, v in imp_entry.entry.items():
                if type(v) == list:
                    self.counter[k].update(v)
                elif type(v) == str:
                    self.counter[k][v] += 1
                else:
                    self.counter[k][k] += 1  # for float or int features, occupy only one column


    def build_attr2idx(self):
        self.attr2idx = defaultdict(dict)  # {Attribute1:set(feat1, feat2, ...), Attribute2: set(feat1, ...), ...}
        self.num_features = 0
        for attr, feat_counter in self.counter.items():
            current_index = 0  # reset for every attribute
            for feat in self.counter[attr]:
                self.attr2idx[attr][feat] = current_index
                current_index += 1
            self.num_features += current_index

    def transform_one(self, doc, ImpressionEntry):
        imp_entry = ImpressionEntry(doc)
        imp_entry.build_entry()
        target = imp_entry.get_target()
        if not target:
            return None
        else:
            return imp_entry.to_feature_vector(self.num_features, self.attr2idx)


    def transform(self, dbname, colname, ImpressionEntry):
        self.col = self.client[dbname][colname]
        n = 0
        total_entires = self.col.find().count()
        matrix = []
        for doc in self.col.find(projection=FEATURE_FIELDS):
            if n % 10000 == 0:
                print('%d/%d' % (n, total_entires))
                print(np.array(matrix))

            n += 1
            vector = self.transform_one(doc, ImpressionEntry)
            if vector:
                matrix.append(vector)

        return np.array(matrix)



if __name__ == "__main__":
    vectorizer = Vectorier()
    vectorizer.fit('Header_Bidding', 'NetworkBackfillImpressions', NetworkBackfillImpressionEntry)
    vectorizer.fit('Header_Bidding', 'NetworkImpressions', NetworkImpressionEntry)
    vectorizer.build_attr2idx()
    pprint(vectorizer.counter)
    pprint(vectorizer.attr2idx)
    pprint(vectorizer.num_features)
    pprint(vectorizer.transform('Header_Bidding', 'NetworkBackfillImpressions', NetworkBackfillImpressionEntry))

