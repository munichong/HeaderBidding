import csv, pandas as pd
from lifelines import CoxPHFitter
from survival_analysis.ImpressionEntry import ImpressionEntry

class NetworkImpressionEntry(ImpressionEntry):
    def __init__(self, doc):
        super().__init__(doc)

    def get_target(self):
        target = []

        ''' Duration '''
        if not self.is_qualified():
            return None
        floor_price = self.get_floor_price()
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

    def is_qualified(self):
        if not self.has_headerbidding():
            return False
        return True
