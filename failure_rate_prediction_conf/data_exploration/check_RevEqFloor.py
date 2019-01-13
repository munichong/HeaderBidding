from pymongo import MongoClient


client = MongoClient()
col = client['Header_Bidding']['NetworkBackfillImpressions']

tol= 0.00001
n_total_adxwon = 0
rev_eq_floor = 0
for doc in col.find():
    revenue = doc['EstimatedBackfillRevenue'] *  1000
    floor = doc['SellerReservePrice']
    n_total_adxwon += 1
    if floor - revenue > tol:
        rev_eq_floor += 1

print(rev_eq_floor, n_total_adxwon, rev_eq_floor / n_total_adxwon)
