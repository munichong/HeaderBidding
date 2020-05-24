import pickle

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from survival_analysis.DataReader import SurvivalData

X_floor_adxwon, Y_maxhd_adxwon = [], []
X_floor_adxlose, Y_minhd_adxlose = [], []
n = 0
for times, events, _, _, min_hbs, max_hbs, _ in SurvivalData(
        *pickle.load(open('../TRAIN_SET.p', 'rb'))).make_sparse_batch(100):
    times, events, min_hbs, max_hbs = shuffle(times, events, min_hbs, max_hbs)
    for t, e, minhb, maxhb in zip(times, events, min_hbs, max_hbs):
        if n > 10000:
            break
        print(n)
        if (
                e == 0
                and maxhb
                and t < maxhb
                and (maxhb - t) / t < 0.01
        ):
            X_floor_adxwon.append(t)
            Y_maxhd_adxwon.append(maxhb)
            n += 1
        elif (
                e == 1
                and minhb
                and t > minhb
                and 0.9 < (t - minhb) / t
        ):
            X_floor_adxlose.append(t)
            Y_minhd_adxlose.append(minhb)
            n += 1

plt.scatter(X_floor_adxwon, Y_maxhd_adxwon, s=0.3)
plt.xlabel("Floor Price")
plt.ylabel("Max Header Bid")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

plt.scatter(X_floor_adxlose, Y_minhd_adxlose, s=0.3, c='r')
plt.xlabel("Floor Price")
plt.ylabel("Min Header Bid")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()
