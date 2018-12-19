import pickle
from sklearn.utils import shuffle
from survival_analysis.DataReader import SurvivalData
from collections import Counter
import matplotlib.pyplot as plt


times_adxwon_1en2, maxhbs_adxwon_1en2 = [], []
times_adxwon_1en3, maxhbs_adxwon_1en3 = [], []
times_adxwon_1en4, maxhbs_adxwon_1en4 = [], []
n = 0
for times, events, _, _, min_hbs, max_hbs, _ in SurvivalData(*pickle.load(open('../TRAIN_SET.p', 'rb'))).make_sparse_batch(100):
    times, events, min_hbs, max_hbs = shuffle(times, events, min_hbs, max_hbs)
    for t, e, minhb, maxhb in zip(times, events, min_hbs, max_hbs):
        if n % 1000000 == 0:
            print(n)
        n += 1

        if e == 1:
            continue

        if t < maxhb and maxhb != 0.0 and (maxhb - t) / t < 0.01:
            times_adxwon_1en2.append(t)
            maxhbs_adxwon_1en2.append(maxhb)

        if t < maxhb and maxhb != 0.0 and (maxhb - t) / t < 0.001:
            times_adxwon_1en3.append(t)
            maxhbs_adxwon_1en3.append(maxhb)

        if t < maxhb and maxhb != 0.0 and (maxhb - t) / t < 0.0001:
            times_adxwon_1en4.append(t)
            maxhbs_adxwon_1en4.append(maxhb)


plt.hist([times_adxwon_1en2, times_adxwon_1en3, times_adxwon_1en4], bins=3000, color=['yellow', 'red', 'blue'])
plt.xlim(0, 20)
plt.ylim(0, 2000)
plt.xlabel("Max Header Bids"
           "\n[YELLOW: floor ((max_hbs-times)/times < 0.01)"
           "\nRED: floor ((max_hbs-times)/times < 0.001)"
           "\nBLUE: floor ((max_hbs-times)/times < 0.0001)]")
plt.ylabel("Count")
plt.show()
