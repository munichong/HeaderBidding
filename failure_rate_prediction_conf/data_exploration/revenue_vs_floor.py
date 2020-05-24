import matplotlib.pyplot as plt
import numpy as np
from pylab import boxplot
from pymongo import MongoClient

client = MongoClient()
col = client['Header_Bidding']['NetworkBackfillImpressions']

x = []
y = []
n = 0
for doc in col.find():
    revenue = doc['EstimatedBackfillRevenue'] * 1000
    floor = doc['SellerReservePrice']
    if revenue <= floor:
        continue
    x.append(floor)
    y.append(revenue)
    # print(floor, revenue)

    if n > 500000:
        break
    n += 1

bins = np.array(list(range(0, 21))) / 10
inds = np.digitize(x, bins)
print(bins)
print(inds)

x_bin = []
y_bin = []
for group in range(min(inds), max(inds)):
    # x_bin.append(x[inds == group])
    y_bin.append(np.array(y)[inds == group])

fig, ax = plt.subplots()
boxplot(y_bin, 0, '')
ax.set_xticklabels(bins)
plt.xlabel("Reserve Price", fontsize=30)
ax.xaxis.set_tick_params(labelsize=20)
plt.ylabel("Impression Revenue", fontsize=30)
ax.yaxis.set_tick_params(labelsize=20)
plt.show()

fig, ax = plt.subplots()
ax.plot(x, y, '.', markersize=0.3, color='black')
plt.xlim(0, 2)
plt.ylim(0, 4)
plt.xlabel("Reserve Price", fontsize=30)
ax.xaxis.set_tick_params(labelsize=20)
plt.ylabel("Impression Revenue", fontsize=30)
ax.yaxis.set_tick_params(labelsize=20)
plt.show()
