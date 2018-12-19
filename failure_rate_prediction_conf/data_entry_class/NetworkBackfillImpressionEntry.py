import pandas as pd
from survival_analysis.data_entry_class.ImpressionEntry import ImpressionEntry


class NetworkBackfillImpressionEntry(ImpressionEntry):
    def __init__(self, doc):
        super().__init__(doc)


    def get_target(self):
        target = []

        ''' Duration '''
        # duration = self.get_floor_price()
        duration = self.get_revenue()
        if not duration:
            return None
        target.append(duration)

        ''' Event '''
        target.append(0)

        return target

    def get_floor_price(self):
        if pd.isnull(self.doc['SellerReservePrice']) or not type(self.doc['SellerReservePrice']) is float:
            self.entry = None
            return None
        return self.doc['SellerReservePrice']

    def get_revenue(self):
        if pd.isnull(self.doc['EstimatedBackfillRevenue']) or not type(self.doc['EstimatedBackfillRevenue']) is float:
            self.entry = None
            return None
        return self.doc['EstimatedBackfillRevenue'] * 1000

    def is_qualified(self):
        # if (
        #         # self.get_revenue() <= 0.0500 and
        #         not self.has_headerbidding()
        # ):
        #     return False
        return True



