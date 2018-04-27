import csv, pandas as pd

HEADER_BIDDING_KEYS = ('mnetbidprice',
                       'mnet_abd',
                       'mnet_fbcpm',
                       'amznbid',
                       'fb_bid_price_cents')
EMPTY = '<EMPTY>'
AMZBID_MAPPING_PATH = '..\PricePoints-3038-display.csv'

class ImpressionEntry:
    def __init__(self, doc):
        self.doc = doc


    def build_entry(self):
        # self.target = []
        self.entry = {}

        # ''' Duration '''
        # if pd.isnull(self.doc['SellerReservePrice']) or not type(self.doc['SellerReservePrice']) is float:
        #     self.entry = None
        #     self.target = None
        #     return
        # self.target.append(self.doc['SellerReservePrice'])
        #
        # ''' Event '''
        # self.target.append(1)


        self.entry['DeviceCategory'] = self.filter_empty_str(self.doc['DeviceCategory'])
        self.entry['MobileDevice'] = self.filter_empty_str(self.doc['MobileDevice'])
        self.entry['Browser'] = self.filter_empty_str(self.doc['Browser']).replace('Any.Any', '').strip()
        self.entry['BandWidth'] = self.filter_empty_str(self.doc['BandWidth'])
        self.entry['OS'] = self.filter_empty_str(self.doc['OS'])
        # self.entry['MobileCarrier'] = self.filter_empty_str(self.doc['MobileCarrier'])

        self.entry['Time'] = self.doc['Time'].hour

        # self.entry['RequestLanguage'] = self.filter_empty_str(self.doc['RequestLanguage'])
        self.entry['Country'] = self.filter_empty_str(self.doc['Country'])
        self.entry['Region'] = self.filter_empty_str(self.doc['Region'])
        # self.entry['Metro'] = self.filter_empty_str(self.doc['Metro'])
        # self.entry['City'] = self.filter_empty_str(self.doc['City'])

        self.entry['RequestedAdUnitSizes'] = self.filter_empty_str(self.doc['RequestedAdUnitSizes']).split('|')
        self.entry['AdPosition'] = self.filter_empty_str(self.doc['AdPosition'])

        for k, v in self.parse_customtargeting(self.doc['CustomTargeting']).items():
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

        feat['trend'] = ct['trend'].lower() if 'trend' in ct else EMPTY
        feat['src'] = ct['src'].lower() if 'src' in ct else EMPTY
        feat['type'] = ct['type'].lower() if 'type' in ct else EMPTY
        feat['ht'] = ct['ht'].lower() if 'ht' in ct else EMPTY

        return feat

    def filter_empty_str(self, string):
        if not string or pd.isnull(string):
            return EMPTY
        return string.lower()

