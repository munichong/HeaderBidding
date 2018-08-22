import bisect
import pandas as pd
import numpy as np
from time import time
from lifelines.utils import _BTree

def c_index(y_pred, events, times):
    df = pd.DataFrame(data={'proba':y_pred, 'event':events, 'time':times})
    df = df.sort_values(by=['time']).reset_index(drop=True)
    earlier_event1_probas = []
    n_total_correct = 0
    n_total_comparable = 0

    event0_rows = df[~df['event'].astype(bool)]
    prev_i = 0
    for cur_i, row in event0_rows.iterrows():
        # print(cur_i)
        # add the event 1 probas in the previous interval into the list
        [bisect.insort_right(earlier_event1_probas, x) for x in df.iloc[prev_i:cur_i, :].proba.tolist()]

        n_correct = len(earlier_event1_probas) - bisect.bisect_right(earlier_event1_probas, row['proba'])
        n_comparable = len(earlier_event1_probas)
        n_total_correct += n_correct
        n_total_comparable += n_comparable
        # print(earlier_event1_probas)
        prev_i = cur_i + 1

        # print()

    return n_total_correct / n_total_comparable if n_total_comparable else None

def c_index1(y_pred, events, times):
    df = pd.DataFrame(data={'proba':y_pred, 'event':events, 'time':times})
    n_total_correct = 0
    n_total_comparable = 0
    df = df.sort_values(by=['time'])
    for i, row in df.iterrows():
        if row['event'] == 1:
            comparable_rows = df[(df['event'] == 0) & (df['time'] > row['time'])]
            n_correct_rows = len(comparable_rows[comparable_rows['proba'] < row['proba']])
            n_total_correct += n_correct_rows
            n_total_comparable += len(comparable_rows)

    return n_total_correct / n_total_comparable if n_total_comparable else None


# start = time()
# r = c_index([0.7, 0.3, 0.67, 0.45, 0.56, 0.8]*100000,
#              [0.0,0.0,1.0,0.0,1.0, 0.0]*100000,
#              [3.1,4.5,6.7,5.2,3.4, 8.1]*100000
#             )
# print(r)
# print(time() - start)
#
# start = time()
# r = c_index1([0.7, 0.3, 0.67, 0.45, 0.56, 0.8]*100000,
#              [0.0,0.0,1.0,0.0,1.0, 0.0]*100000,
#              [3.1,4.5,6.7,5.2,3.4, 8.1]*100000
#             )
# print(r)
# print(time() - start)

# def c_index2(y_pred, events, times):
#
#     df = pd.DataFrame(data={'proba':y_pred, 'event':events, 'time':times})
#     died_mask = df['event'].astype(bool)
#     died_truth = df[died_mask]
#     ix = np.argsort(died_truth)
#     print(ix)
