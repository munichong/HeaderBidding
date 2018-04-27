import pandas as pd
from datetime import datetime

class NetworkBackfillImpressionEntry():
    def __init__(self, doc):
        self.doc = doc

    def build_entry(self):
        entry = []

        ''' Duration '''
        if pd.isnull(self.doc['SellerReservePrice']) or not type(self.doc['SellerReservePrice']) is float:
            return None
        entry.append(self.doc['SellerReservePrice'])

        ''' Event '''
        entry.append(1)


        entry.append(self.filter_empty_str(self.doc['DeviceCategory']))
        entry.append(self.filter_empty_str(self.doc['MobileDevice']))
        entry.append(self.filter_empty_str(self.doc['Browser']).replace('Any.Any', '').strip())
        entry.append(self.filter_empty_str(self.doc['BandWidth']))
        entry.append(self.filter_empty_str(self.doc['OS']))
        entry.append(self.filter_empty_str(self.doc['MobileCarrier']))

        entry.append(self.doc['Time'].hour)

        entry.append(self.filter_empty_str(self.doc['RequestLanguage']))
        entry.append(self.filter_empty_str(self.doc['Country']))
        entry.append(self.filter_empty_str(self.doc['Region']))
        # entry.append(self.filter_empty_str(self.doc['Metro']))
        # entry.append(self.filter_empty_str(self.doc['City']))









    def filter_empty_str(self, string):
        if not string or pd.isnull(string):
            return '<EMPTY>'
        return string.lower()
