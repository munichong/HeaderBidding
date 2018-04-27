from pymongo import MongoClient
from survival_analysis.NetworkBackfillImpressionEntries import NetworkBackfillImpressionEntries

DBNAME = 'Header_Bidding'
FEATURE_FIELDS = []

client = MongoClient()


def filter_rows(colname=None):
    col = client[DBNAME][colname]
    data = []  # stores tuples
    for doc in col.find(projection=FEATURE_FIELDS):
        NetworkBackfillImpressionEntries(doc)






all_data = []
all_data.extend(filter_rows('NetworkBackfillImpressions'))
all_data.extend(filter_rows('NetworkImpressions'))