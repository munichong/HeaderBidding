import pandas as pd
from datetime import datetime

HEADER_BIDDING_KEYS = ('mnetbidprice',
                       'mnet_abd',
                       'mnet_fbcpm',
                       'amznbid',
                       'fb_bid_price_cents')
EMPTY = '<EMPTY>'

class NetworkBackfillImpressionEntry:
    def __init__(self, doc):
        self.doc = doc

    def build_entry(self):
        self.target = []
        self.entry = {}

        ''' Duration '''
        if pd.isnull(self.doc['SellerReservePrice']) or not type(self.doc['SellerReservePrice']) is float:
            return None
        self.target.append(self.doc['SellerReservePrice'])

        ''' Event '''
        self.target.append(1)

        self.entry['DeviceCategory'] = self.filter_empty_str(self.doc['DeviceCategory'])
        self.entry['MobileDevice'] = self.filter_empty_str(self.doc['MobileDevice'])
        self.entry['Browser'] = self.filter_empty_str(self.doc['Browser']).replace('Any.Any', '').strip()
        self.entry['BandWidth'] = self.filter_empty_str(self.doc['BandWidth'])
        self.entry['OS'] = self.filter_empty_str(self.doc['OS'])
        self.entry['MobileCarrier'] = self.filter_empty_str(self.doc['MobileCarrier'])

        self.entry['Time'] = self.doc['Time'].hour

        self.entry['RequestLanguage'] = self.filter_empty_str(self.doc['RequestLanguage'])
        self.entry['Country'] = self.filter_empty_str(self.doc['Country'])
        self.entry['Region'] = self.filter_empty_str(self.doc['Region'])
        # self.entry['Metro'] = self.filter_empty_str(self.doc['Metro'])
        # self.entry['City'] = self.filter_empty_str(self.doc['City'])

        self.entry['RequestedAdUnitSizes'] = self.filter_empty_str(self.doc['RequestedAdUnitSizes']).split('|')
        self.entry['AdPosition'] = self.filter_empty_str(self.doc['AdPosition'])

        for k, v in self.parse_customtargeting(self.doc['CustomTargeting']):
            self.entry[k] = v


    def parse_customtargeting(self, ct):
        feat = {}

        feat['displaychannel'] = ct['displaychannel'] if 'displaychannel' in ct else EMPTY
        feat['displaysection'] = ct['displaysection'] if 'displaysection' in ct else EMPTY


        if 'channel' in ct:
            feat['channel'] = ct['channel'] if type(ct['channel']) == list else [ct['channel']]
        else:
            feat['channel'] = []

        if 'section' in ct:
            feat['section'] = ct['section'] if type(ct['section']) == list else [ct['section']]
        else:
            feat['section'] = []

        for hd_key in HEADER_BIDDING_KEYS:
            feat[hd_key] = float(ct[hd_key]) if hd_key in ct else 0.0


        feat['trend'] = ct['trend'].lower() if 'trend' in ct else EMPTY
        feat['src'] = ct['src'].lower() if 'src' in ct else EMPTY
        feat['type'] = ct['type'].lower() if 'type' in ct else EMPTY
        feat['ht'] = ct['ht'].lower() if 'ht' in ct else EMPTY

        return feat

    def filter_empty_str(self, string):
        if not string or pd.isnull(string):
            return EMPTY
        return string.lower()

    def to_cox_vector(self):  # vectorize
        pass