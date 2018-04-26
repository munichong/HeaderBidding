from pymongo import MongoClient

DBNAME = 'Header_Bidding'
COLNAME = 'NetworkBackfillImpressions'
FEATURE_FIELDS = []


class NetworkBackfillImpressionEntries:
    def __init__(self):
        client = MongoClient()
        self.col = client[DBNAME][COLNAME]

    def retrieve_entries(self):
        self.data = []  # stores tuples
        for doc in self.col.find(projection=FEATURE_FIELDS):
            pass


    def to_cox_problem(self):  # vectorize
        pass


