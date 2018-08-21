import pandas as pd

def c_index(y_pred, events, times):
    df = pd.DataFrame(data={'proba':y_pred, 'event':events, 'time':times})
    n_total_correct = 0
    n_total_comparable = 0
    for i, row in df.iterrows():
        # print(row)

        if row['event'] == 0:
            comparable_rows = df[(df.index > i) & (df['event'] == 1) & (df['time'] < row['time'])]
            n_comparable_rows = len(comparable_rows)
            n_correct_rows = len(comparable_rows[comparable_rows['proba'] > row['proba']])
            # print(comparable_rows[comparable_rows['proba'] > row['proba']])
        else:
            comparable_rows = df[(df.index > i) & (df['event'] == 0) & (df['time'] > row['time'])]
            n_comparable_rows = len(comparable_rows)
            n_correct_rows = len(comparable_rows[comparable_rows['proba'] < row['proba']])
            # print(comparable_rows[comparable_rows['proba'] > row['proba']])
        # print()
        n_total_correct += n_correct_rows
        n_total_comparable += n_comparable_rows



        # print()
    return n_total_correct / n_total_comparable if n_total_comparable else 0.0


# c = c_index([0.1, 0.3, 0.67, 0.45, 0.56], [1.0,0.0,1.0,0.0,1.0], [3.1,4.5,6.7,5.2,3.4])
# print(c)

