from pymongo import MongoClient

client = MongoClient()
col = client['Header_Bidding']['NetworkBackfillImpressions']

n_total_adxwon = 0
n_less_0_05 = 0
n_less_0_05_N_floor = 0
for doc in col.find():
    revenue = doc['EstimatedBackfillRevenue'] * 1000
    floor = doc['SellerReservePrice']
    n_total_adxwon += 1
    if revenue < 0.05:
        n_less_0_05 += 1
        if revenue < floor:
            n_less_0_05_N_floor += 1
            # print(floor, revenue)

print(n_less_0_05, n_total_adxwon, n_less_0_05 / n_total_adxwon)
print(n_less_0_05_N_floor, n_less_0_05_N_floor / n_less_0_05)
