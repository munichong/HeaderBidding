import csv, pandas as pd
from lifelines import CoxPHFitter
from survival_analysis.ImpressionEntry import ImpressionEntry

HEADER_BIDDING_KEYS = ('mnetbidprice',
                       'mnet_abd',
                       'mnet_fbcpm',
                       'amznbid',
                       'fb_bid_price_cents')
EMPTY = '<EMPTY>'
AMZBID_MAPPING_PATH = '..\PricePoints-3038-display.csv'

class NetworkImpressionEntry(ImpressionEntry):
    def __init__(self, doc):
        super().__init__(doc)

    def get_target(self):
        target = []

        ''' Duration '''
        floor_price = self.get_floor_price()
        if not floor_price:
            return None
        target.append(floor_price)

        ''' Event '''
        target.append(0)

        return target

    def get_floor_price(self):
        highest_header_bid = self.get_highest_header_bid()
        if not highest_header_bid:
            return None
        return self.to_closest_5cents(highest_header_bid)

    def get_highest_header_bid(self):
        header_bids = self.get_headerbidding()
        if not header_bids:
            return None
        return max(header_bids)

    def to_closest_5cents(self, num):
        return num - (num % 0.05)
