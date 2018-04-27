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

    def load_amznbid_price_mapping(self):
        self.amzbid_mapping = {}
        with open(AMZBID_MAPPING_PATH) as infile:
            csv_reader = csv.reader(infile, delimiter=',')
            next(csv_reader)
            for line in csv_reader:
                self.amzbid_mapping[line[-1]] = float(line[-2].replace('$', '').strip())

    def get_headerbidding(self, ct):
        header_bids = []
        for hd_key in HEADER_BIDDING_KEYS:
            if hd_key == 'fb_bid_price_cents':
                header_bids[hd_key] = float(ct[hd_key]) / 100 if hd_key in ct else 0.0
            elif hd_key == 'amznbid':
                header_bids[hd_key] = self.amzbid_mapping[ct[hd_key]] if hd_key in ct and ct[hd_key] in self.amzbid_mapping else 0.0
            else:
                header_bids[hd_key] = float(ct[hd_key]) if hd_key in ct else 0.0
        return header_bids

    def get_target(self):
        self.target = []

        ''' Duration '''
        if pd.isnull(self.doc['SellerReservePrice']) or not type(self.doc['SellerReservePrice']) is float:
            self.entry = None
            self.target = None
            return
        self.target.append(self.doc['SellerReservePrice'])

        ''' Event '''
        self.target.append(1)
