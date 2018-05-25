from pymongo import MongoClient


HEADER_BIDDING_KEYS = ('mnetbidprice',
                        'mnet_abd',
                        'mnet_fbcpm',
                        'amznbid',
                        'crt_pb',
                        'fb_bid_price_cents')

client = MongoClient()
col = client['Header_Bidding']['NetworkImpressions']

n_total_hdwon = 0
n_without_hd = 0
for doc in col.find():
    n_total_hdwon += 1
    if any(k in HEADER_BIDDING_KEYS for k, v in doc['CustomTargeting'].items()):
        continue
    n_without_hd += 1
    print(doc)

print(n_without_hd, n_total_hdwon, n_without_hd / n_total_hdwon)

