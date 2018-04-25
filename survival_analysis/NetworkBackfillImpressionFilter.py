from pymongo import MongoClient

DBNAME = 'Header_Bidding'
COLNAME = 'NetworkBackfillImpressions'
FEATURE_FIELDS = []


class NetworkBackfillImpressionFilter:
    def __init__(self):
        client = MongoClient()
        self.col = client[DBNAME][COLNAME]

    def filter(self):
        self.data = []  # stores tuples
        for doc in self.col.find(projection=FEATURE_FIELDS):

            '''
            filtered = df.groupby('positions')['r vals'].filter(lambda x: len(x) >= 3)
            df[df['r vals'].isin(filtered)]
            '''
