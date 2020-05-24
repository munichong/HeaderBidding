import numpy as np
import pickle

from survival_analysis.DataReader import SurvivalData

n_hb_overestimate = 0
n_hb_underestimate = 0
n_hb_adxwon = 0
n_hb_adxlose = 0
n_maxhb_l_t = 0
n_event0 = 0
n_event1 = 0
n_event1_maxhb_l_t = 0
n_minhb_g_t = 0
n_meanhb_g_t = 0
n_medianhb_g_t = 0
n_maxhb_g_t = 0
n_minhb_l_t = 0
n_medianhb_l_t = 0
n_meanhb_l_t = 0
for times, events, _, _, min_hbs, max_hbs, hbs, _ in SurvivalData(
        *pickle.load(open('../TRAIN_SET.p', 'rb'))).make_sparse_batch(100000):
    for t, e, minhb, maxhb, hb in zip(times, events, min_hbs, max_hbs, hbs):
        if maxhb == 0 or minhb == 0:
            continue

        if len(hb) < 2:
            continue

        if e == 0:
            n_event0 += 1
            if minhb > t:
                n_minhb_g_t += 1
            if np.mean(hb) > t:
                n_meanhb_g_t += 1
            if np.median(hb) > t:
                n_medianhb_g_t += 1
            if maxhb > t:
                n_maxhb_g_t += 1

        else:
            n_event1 += 1
            if minhb < t:
                n_minhb_l_t += 1
            if np.mean(hb) < t:
                n_meanhb_l_t += 1
            if np.median(hb) < t:
                n_medianhb_l_t += 1
            if maxhb < t:
                n_maxhb_l_t += 1
        """
        if maxhb > t:
            if e == 1:
                n_hb_overestimate += 1
            else:
                n_hb_adxwon += 1
            n_maxhb_g_t += 1

        if maxhb < t:
            if e == 0:
                n_hb_underestimate += 1
            n_maxhb_l_t += 1

        # if e == 1 and maxhb < t:
        #     n_event1_maxhb_l_t += 1

        if minhb < t and e == 1:
            n_hb_adxlose += 1
        


print(n_hb_overestimate, n_hb_underestimate, n_maxhb_g_t, n_maxhb_l_t, n_event0, n_event1)
print("To prove max_hb is not a reliable failure point indicator:")
print(n_hb_overestimate / n_maxhb_g_t * 100, "percent")  # the higher the less reliable
print(n_hb_underestimate / n_maxhb_l_t * 100, "percent")  # the higher the less reliable
print("To prove hb reg may make sense:")
print(n_hb_adxwon / n_event0 * 100, "percent")  # the higher the better
print(n_hb_adxlose / n_event1 * 100, "percent")  # the higher the better
"""

# print()
# print(n_event1_maxhb_l_t)

print(
    "When event is 0, %.4f%% min hb > floor, %.4f%% mean hb > floor, %.4f%% median hb > floor, %.4f%% max hb > floor" %
    (n_minhb_g_t / n_event0 * 100, n_meanhb_g_t / n_event0 * 100, n_medianhb_g_t / n_event0 * 100,
     n_maxhb_g_t / n_event0 * 100))
print(
    "When event is 1, %.4f%% min hb < floor, %.4f%% mean hb < floor, %.4f%% median hb < floor, %.4f%% max hb < floor" %
    (n_minhb_l_t / n_event1 * 100, n_meanhb_l_t / n_event1 * 100, n_medianhb_l_t / n_event1 * 100,
     n_maxhb_l_t / n_event1 * 100))
