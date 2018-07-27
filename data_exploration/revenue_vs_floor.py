from pymongo import MongoClient
import matplotlib.pyplot as plt


client = MongoClient()
col = client['Header_Bidding']['NetworkBackfillImpressions']


x = []
y = []
for doc in col.find():
    revenue = doc['EstimatedBackfillRevenue'] *  1000
    floor = doc['SellerReservePrice']
    if revenue == floor:
        continue
    x.append(floor)
    y.append(revenue)

plt.plot(x, y, 'o', color='black')