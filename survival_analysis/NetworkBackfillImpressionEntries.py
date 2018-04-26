from pymongo import MongoClient

DBNAME = 'Header_Bidding'
COLNAME = 'NetworkBackfillImpressions'
FEATURE_FIELDS = ['URIs_pageno', 'NaturalIDs', 'RefererURL', 'UserId',
                  'DeviceCategory', 'MobileDevice', 'Browser', 'BandWidth', 'OS', 'MobileCarrier',
                  'SellerReservePrice', 'EstimatedBackfillRevenue',
                  'TimeUsec',
                  'RequestLanguage', 'Country', 'Region', 'Metro', 'City', 'AudienceSegmentIds',
                  'RequestedAdUnitSizes', 'CreativeSize', 'AdPosition'
                  'CustomTargeting', ]


class NetworkBackfillImpressionEntries:
    def __init__(self):
        client = MongoClient()
        self.col = client[DBNAME][COLNAME]

    def retrieve_entries(self):
        self.data = []  # stores tuples
        for doc in self.col.find(projection=FEATURE_FIELDS):




    def to_cox_problem(self):  # vectorize
        pass


