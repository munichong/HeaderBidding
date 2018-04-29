import csv, pandas as pd
from datetime import datetime
from survival_analysis.ImpressionEntry import ImpressionEntry

HEADER_BIDDING_KEYS = ('mnetbidprice',
                       'mnet_abd',
                       'mnet_fbcpm',
                       'amznbid',
                       'fb_bid_price_cents')
EMPTY = '<EMPTY>'
AMZBID_MAPPING_PATH = '..\PricePoints-3038-display.csv'


class NetworkBackfillImpressionEntry(ImpressionEntry):
    def __init__(self, doc):
        super().__init__(doc)
        self.load_amznbid_price_mapping()

    def get_target(self):
        target = []

        ''' Duration '''
        if pd.isnull(self.doc['SellerReservePrice']) or not type(self.doc['SellerReservePrice']) is float:
            self.entry = None
            target = None
            return None
        target.append(self.doc['SellerReservePrice'])

        ''' Event '''
        target.append(1)

        return target

    def to_feature_vector(self, n_feats, attr2idx):
        vector = [0] * n_feats
        for attr, feats in self.entry.items():
            if type(feats) == list:
                for f in feats:
                    vector[attr2idx[attr][f]] = 1
            elif type(feats) == str:
                vector[attr2idx[attr][feats]] = 1
            else:
                vector[attr2idx[attr][attr]] = feats
        return vector + self.get_headerbidding()

