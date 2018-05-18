import os, csv, pickle
from pymongo import MongoClient
from pprint import pprint
from survival_analysis.data_entry_class.NetworkBackfillImpressionEntry import NetworkBackfillImpressionEntry
from survival_analysis.data_entry_class.NetworkImpressionEntry import NetworkImpressionEntry
from survival_analysis.data_entry_class.ImpressionEntry import HEADER_BIDDING_KEYS
from collections import defaultdict, Counter


FEATURE_FIELDS = ['URIs_pageno', 'NaturalIDs', 'RefererURL', 'UserId',
                  'DeviceCategory', 'MobileDevice', 'Browser', 'BandWidth', 'OS', 'MobileCarrier',
                  'SellerReservePrice', 'EstimatedBackfillRevenue',
                  'Time',
                  'RequestLanguage', 'Country', 'Region', 'Metro', 'City',
                  'RequestedAdUnitSizes', 'AdPosition',
                  'CustomTargeting', ]


class Vectorizer:
    def __init__(self):
        self.client = MongoClient()
        self.counter = defaultdict(Counter)  # {Attribute1:set(feat1, feat2, ...), Attribute2: set(feat1, ...), ...}

    def fit(self, dbname, colname, ImpressionEntry):
        '''count unique attributes'''
        self.col = self.client[dbname][colname]

        # STOP = False
        n = 0
        total_entires = self.col.find().count()
        for doc in self.col.find(projection=FEATURE_FIELDS):
            if n % 1000000 == 0:
                print('%d/%d' % (n, total_entires))
                # if STOP:
                #     break
                # STOP = True
            n += 1

            imp_entry = ImpressionEntry(doc)
            imp_entry.build_entry()

            if not imp_entry.is_qualified():
                continue

            # if imp_entry.get_reserveprice() > imp_entry.get_revenue():
            #     # max(imp_entry.get_headerbidding())
            #     print(imp_entry.get_reserveprice(), imp_entry.get_revenue(), imp_entry.doc['CustomTargeting'])
            #     print(imp_entry.doc)
            #     print()

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
            for feat in self.counter[attr]:
                self.attr2idx[attr][feat] = self.num_features
                self.num_features += 1

    def transform_one(self, doc, ImpressionEntry):
        imp_entry = ImpressionEntry(doc)
        imp_entry.build_entry()
        target = imp_entry.get_target()
        if not imp_entry.is_qualified() or not target:
            return None

        # return target + imp_entry.to_full_feature_vector(self.num_features, self.attr2idx)
        return target + imp_entry.to_sparse_feature_vector(self.num_features, self.attr2idx)


    def transform(self, dbname, colname, ImpressionEntry):
        self.col = self.client[dbname][colname]
        n = 0
        total_entries = self.col.find().count()
        matrix = []
        for doc in self.col.find(projection=FEATURE_FIELDS):
            if n % 1000000 == 0:
                print('%d/%d' % (n, total_entries))
                yield matrix
                matrix.clear()

            n += 1
            vector = self.transform_one(doc, ImpressionEntry)
            if vector:
                matrix.append(vector)

        yield matrix
        matrix.clear()


def output_vector_files(path, colname, ImpressionEntry):
    with open(path, 'a', newline='\n') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow([vectorizer.num_features + len(HEADER_BIDDING_KEYS)])  # the number of features WITH header bidding BUT WITHOUT 'duration' and 'event'
        for mat in vectorizer.transform('Header_Bidding', colname, ImpressionEntry):
            writer.writerows(mat)


if __name__ == "__main__":
    vectorizer = Vectorizer()
    vectorizer.fit('Header_Bidding', 'NetworkBackfillImpressions', NetworkBackfillImpressionEntry)
    pprint(vectorizer.counter)
    vectorizer.fit('Header_Bidding', 'NetworkImpressions', NetworkImpressionEntry)
    vectorizer.build_attr2idx()
    pprint(vectorizer.counter)
    pprint(vectorizer.attr2idx)
    pprint(vectorizer.num_features)

    pickle.dump(vectorizer.counter, open("../counter.dict", "wb"))
    pickle.dump(vectorizer.attr2idx, open("../attr2idx.dict", "wb"))  # the dict does not contain header bidding.

    try:
        os.remove('../Vectors_adxwon.csv')
        os.remove('../Vectors_adxlose.csv')
    except OSError:
        pass

    output_vector_files('../Vectors_adxwon.csv', 'NetworkBackfillImpressions', NetworkBackfillImpressionEntry)
    output_vector_files('../Vectors_adxlose.csv', 'NetworkImpressions', NetworkImpressionEntry)
