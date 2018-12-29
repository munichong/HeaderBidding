import os, csv, pickle
from pymongo import MongoClient
from pprint import pprint
from failure_rate_prediction_conf.data_entry_class.NetworkBackfillImpressionEntry import NetworkBackfillImpressionEntry
from failure_rate_prediction_conf.data_entry_class.NetworkImpressionEntry import NetworkImpressionEntry
from failure_rate_prediction_conf.data_entry_class.ImpressionEntry import HEADER_BIDDING_KEYS
from collections import defaultdict, Counter


FEATURE_FIELDS = ['URIs_pageno', 'NaturalIDs', 'RefererURL', 'UserId',
                  'DeviceCategory', 'MobileDevice', 'Browser', 'BandWidth', 'OS', 'MobileCarrier',
                  'SellerReservePrice', 'EstimatedBackfillRevenue',
                  'Time',
                  'RequestLanguage', 'Country', 'Region', 'Metro', 'City',
                  'RequestedAdUnitSizes', 'AdPosition',
                  'CustomTargeting', ]
MIN_OCCURRENCE = 5
MIN_OCCURRENCE_SYMBOL = '<RARE>'

class Vectorizer:
    def __init__(self):
        self.client = MongoClient()
        self.counter = defaultdict(Counter)  # {Attribute1:Counter<features>, Attribute2:Counter<features>, ...}

    def fit(self, dbname, colname, ImpressionEntry):
        '''count unique attributes'''
        self.col = self.client[dbname][colname]

        # STOP = False
        n = 0
        total_entries = self.col.find().count()
        for doc in self.col.find(projection=FEATURE_FIELDS):
            if n % 1000000 == 0:
                print('%d/%d (%.2f%%)' % (n, total_entries, n / total_entries * 100))
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

            for k, v in imp_entry.entry.items():  # iterate all <fields:feature>
                if type(v) == list:
                    self.counter[k].update(v)
                elif type(v) == str:
                    self.counter[k][v] += 1
                else:
                    self.counter[k][k] += 1  # for float or int features, occupy only one column


    def build_attr2idx(self):
        self.attr2idx = defaultdict(dict)  # {Attribute1: dict(feat1:i, ...), Attribute2: dict(feat1:i, ...), ...}
        self.num_features = 0
        for attr, feat_counter in self.counter.items():
            most_common_feature = self.counter[attr].most_common(1)[0][0]
            for feat in self.counter[attr]:
                if feat == most_common_feature:  # skip the most common feature in each attribute to avoid dummy variable trap
                    continue

                if self.counter[attr][feat] < MIN_OCCURRENCE:
                    if MIN_OCCURRENCE_SYMBOL in self.attr2idx[attr]:
                        continue
                    feat = MIN_OCCURRENCE_SYMBOL

                self.attr2idx[attr][feat] = self.num_features
                self.num_features += 1

    def transform_one(self, doc, ImpressionEntry):
        imp_entry = ImpressionEntry(doc)
        imp_entry.build_entry()
        target = imp_entry.get_target()
        if not imp_entry.is_qualified() or not target:
            return None, None
        header_bids = imp_entry.to_sparse_headerbids()
        # return target + imp_entry.to_full_feature_vector(self.num_features, self.attr2idx)
        return target + imp_entry.to_sparse_feature_vector(self.attr2idx, self.counter), header_bids


    def transform(self, dbname, colname, ImpressionEntry):
        self.col = self.client[dbname][colname]
        n = 0
        total_entries = self.col.find().count()
        matrix = []
        header_bids = []
        for doc in self.col.find(projection=FEATURE_FIELDS):
            if n % 1000000 == 0:
                print('%d/%d (%.2f%%)' % (n, total_entries, n / total_entries * 100))
                yield matrix, header_bids
                matrix.clear()
                header_bids.clear()

            n += 1
            feat_vector, hbs = self.transform_one(doc, ImpressionEntry)

            if feat_vector:
                matrix.append(feat_vector)
                header_bids.append(hbs)

        yield matrix, header_bids
        matrix.clear()
        header_bids.clear()


def output_vector_files(featfile_path, hbfile_path, colname, ImpressionEntry):
    with open(featfile_path, 'a', newline='\n') as outfile_feat, open(hbfile_path, 'a', newline='\n') as outfile_hb:
        writer_feat = csv.writer(outfile_feat, delimiter=',')
        writer_hb = csv.writer(outfile_hb, delimiter=',')
        writer_feat.writerow([vectorizer.num_features])  # the number of features WITH header bidding BUT WITHOUT 'duration', 'event', and header bids
        writer_hb.writerow([len(HEADER_BIDDING_KEYS)])
        for mat, hbs in vectorizer.transform('Header_Bidding', colname, ImpressionEntry):
            writer_feat.writerows(mat)
            writer_hb.writerows(hbs)



if __name__ == "__main__":
    vectorizer = Vectorizer()
    print('Fitting NetworkBackfillImpressions...')
    vectorizer.fit('Header_Bidding', 'NetworkBackfillImpressions', NetworkBackfillImpressionEntry)
    pprint(vectorizer.counter)
    print()
    print('Fitting NetworkImpressions...')
    vectorizer.fit('Header_Bidding', 'NetworkImpressions', NetworkImpressionEntry)
    vectorizer.build_attr2idx()
    pprint(vectorizer.counter)
    pprint(vectorizer.attr2idx)
    pprint(vectorizer.num_features)

    '''
    counter does NOT contain header bidding.
    counter contains the most common feature in each attribute
    '''
    pickle.dump(vectorizer.counter, open("output/counter.dict", "wb"))
    '''
    attr2idx does NOT contain header bidding.
    attr2idx does NOT contain the most common feature in each attribute
    '''
    pickle.dump(vectorizer.attr2idx, open("output/attr2idx.dict", "wb"))

    try:
        os.remove('output/FeatVec_adxwon.csv')
        os.remove('output/FeatVec_adxlose.csv')
        os.remove('output/HeaderBids_adxwon.csv')
        os.remove('output/HeaderBids_adxlose.csv')
    except OSError:
        pass

    output_vector_files('output/FeatVec_adxwon.csv',
                        'output/HeaderBids_adxwon.csv',
                        'NetworkBackfillImpressions',
                        NetworkBackfillImpressionEntry)
    output_vector_files('output/FeatVec_adxlose.csv',
                        'output/HeaderBids_adxlose.csv',
                        'NetworkImpressions',
                        NetworkImpressionEntry)
