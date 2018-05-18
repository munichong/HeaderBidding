import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

class UnivariateLogisticRegression:

    def __init__(self):
        self.lr = LogisticRegression(penalty='l2', C=1.0)

    def fit(self, X, y):
        return self.lr.fit(X, y)

    def predict_proba(self, X):
        return self.lr.predict_proba(X)

    def predict(self, X):
        return self.lr.predict(X)

    def evaluate(self, X, y_bin_true):
        y_prob_pred = self.predict_proba(X)
        y_bin_pred = self.predict(X)
        return log_loss(y_bin_true, y_prob_pred), roc_auc_score(y_bin_true, y_bin_pred), accuracy_score(y_bin_true, y_bin_pred)



if __name__ == '__main__':
    def read_data(file_path):
        return pickle.load(open(file_path, 'rb'))

    def expand_dims(data_list, axis):
        return [np.expand_dims(d, axis=axis) for d in data_list]

    baseline = UnivariateLogisticRegression()

    times_train, events_train = expand_dims(read_data('../Vectors_train.p')[:2], axis=1)
    times_val, events_val = expand_dims(read_data('../Vectors_val.p')[:2], axis=1)
    times_test, events_test = expand_dims(read_data('../Vectors_test.p')[:2], axis=1)

    baseline.fit(np.array(times_train), np.array(events_train))
    print(baseline.evaluate(np.array(times_train), np.array(events_train)))
    print(baseline.evaluate(np.array(times_val), np.array(events_val)))
    print(baseline.evaluate(np.array(times_test), np.array(events_test)))
