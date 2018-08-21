import pandas as pd
import numpy as np
from time import time

def c_index(y_pred, events, times):

    df = pd.DataFrame(data={'proba':y_pred, 'event':events, 'time':times})
    df = df.sort_values(by=['time'])

    n_total_correct = 0
    n_total_comparable = 0

    def cal(row):
        nonlocal n_total_correct, n_total_comparable
        comparable_rows = df[(df['event'] == 0) & (df['time'] > row.time)]
        n_correct_rows = comparable_rows[comparable_rows['proba'] < row.proba].shape[0]
        n_total_correct += n_correct_rows
        n_total_comparable += comparable_rows.shape[0]

    died_mask = df['event'].astype(bool)
    map(cal, df[died_mask])

    return n_total_correct / n_total_comparable if n_total_comparable else None

# def c_index1(y_pred, events, times):
#     df = pd.DataFrame(data={'proba':y_pred, 'event':events, 'time':times})
#     n_total_correct = 0
#     n_total_comparable = 0
#     df = df.sort_values(by=['time'])
#     for i, row in df.iterrows():
#         if row['event'] == 1:
#             comparable_rows = df[(df['event'] == 0) & (df['time'] > row['time'])]
#             n_correct_rows = len(comparable_rows[comparable_rows['proba'] < row['proba']])
#             n_total_correct += n_correct_rows
#             n_total_comparable += len(comparable_rows)
#
#     return n_total_correct / n_total_comparable if n_total_comparable else None
#
# start = time()
# c = c_index([0.1, 0.3, 0.67, 0.45, 0.56]*100000, [1.0,0.0,1.0,0.0,1.0]*100000, [3.1,4.5,6.7,5.2,3.4]*100000)
# print(c)
# print(time() - start)
#
# start = time()
# c = c_index1([0.1, 0.3, 0.67, 0.45, 0.56]*100000, [1.0,0.0,1.0,0.0,1.0]*100000, [3.1,4.5,6.7,5.2,3.4]*100000)
# print(c)
# print(time() - start)
#
# def c_index2(y_pred, events, times):
#
#     df = pd.DataFrame(data={'proba':y_pred, 'event':events, 'time':times})
#     died_mask = df['event'].astype(bool)
#     died_truth = df[died_mask]
#     ix = np.argsort(died_truth)
#     print(ix)
