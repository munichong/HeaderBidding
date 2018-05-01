import csv, pandas as pd
from datetime import datetime
from survival_analysis.ImpressionEntry import ImpressionEntry


class NetworkBackfillImpressionEntry(ImpressionEntry):
    def __init__(self, doc):
        super().__init__(doc)


    def get_target(self):
        target = []

        ''' Duration '''
        reserve_price = self.get_reserveprice()
        if not reserve_price:
            return None
        target.append(reserve_price)

        ''' Event '''
        target.append(1)

        return target

    def get_reserveprice(self):
        if pd.isnull(self.doc['SellerReservePrice']) or not type(self.doc['SellerReservePrice']) is float:
            self.entry = None
            return None
        return self.doc['SellerReservePrice']

    def is_qualified(self):
        if self.get_reserveprice() <= 0.050001 and not self.has_headerbidding():
            return False
        return True



