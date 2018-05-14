import csv
import tensorflow as tf
import numpy as np
from scipy.sparse import coo_matrix


class SurvivalData:

    def __init__(self, file_path, num_epochs, num_features):
        self.num_features = num_features
        self.num_data = sum(1 for _ in csv.reader(open(file_path))) - 1  # minus the header line
        self.times, self.events, self.sparse_features = self.read_data(file_path)

    def read_data(self, file_path):
        times, events = [], []
        row_indices, col_indices, values = [], [], []
        num_rows = 0
        with open(file_path) as infile:
            csv_reader = csv.reader(infile, delimiter=',')
            for row_index, row in enumerate(csv_reader):
                num_rows += 1
                times.append(float(row[0]))
                events.append(int(row[1]))
                for node in row[2:]:
                    col_index, val = node.split(':')
                    row_indices.append(row_index)
                    col_indices.append(int(col_index))
                    values.append(float(val))
        return np.array(times), \
               np.array(events), \
               coo_matrix((values, (row_indices, col_indices)), shape=(num_rows, self.num_features))


if __name__ == "__main__":
    s = SurvivalData('../Vectors_train.csv', 30, 6026)
    print(s.times)
    print(s.events)
    print(s.sparse_features)

