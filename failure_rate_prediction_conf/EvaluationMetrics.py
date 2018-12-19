import bisect
import pandas as pd

def c_index(y_pred, events, times):
    df = pd.DataFrame(data={'proba':y_pred, 'event':events, 'time':times})
    df = df.sort_values(by=['time']).reset_index(drop=True)
    earlier_event1_probas = []
    n_total_correct = 0
    n_total_comparable = 0

    event0_rows = df[~df['event'].astype(bool)]
    prev_i = 0
    for cur_i, row in event0_rows.iterrows():
        # add the event 1 probas in the previous interval into the list
        [bisect.insort_right(earlier_event1_probas, x) for x in df.iloc[prev_i:cur_i, :].proba.tolist()]

        n_correct = len(earlier_event1_probas) - bisect.bisect_right(earlier_event1_probas, row['proba'])
        n_comparable = len(earlier_event1_probas)
        n_total_correct += n_correct
        n_total_comparable += n_comparable
        prev_i = cur_i + 1

    return n_total_correct / n_total_comparable if n_total_comparable else None