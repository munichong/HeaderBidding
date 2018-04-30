import csv, pandas as pd
from datetime import datetime
from survival_analysis.ImpressionEntry import ImpressionEntry


class NetworkBackfillImpressionEntry(ImpressionEntry):
    def __init__(self, doc):
        super().__init__(doc)


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



