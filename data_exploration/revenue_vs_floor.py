from pymongo import MongoClient
import matplotlib.pyplot as plt


client = MongoClient()
col = client['Header_Bidding']['NetworkBackfillImpressions']


x = []
y = []
n = 0
for doc in col.find():
    revenue = doc['EstimatedBackfillRevenue'] *  1000
    floor = doc['SellerReservePrice']
    if revenue <= floor:
        continue
    x.append(floor)
    y.append(revenue)
    print(floor, revenue)

    if n > 10000:
        break
    n += 1

plt.plot(x, y, 'o', markersize=1, color='black')
plt.xlim(0, 2)
plt.ylim(0, 2)
plt.xlabel("Reserve Price")
plt.ylabel("Impression Revenue")
plt.show()