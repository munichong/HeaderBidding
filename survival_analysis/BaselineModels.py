import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

class UnivariateLogisticRegression:
    # def get_data(self, file_path):
    #     times, events, _ = pickle.load(open(file_path, 'rb'))
    #     return times. events
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
    baseline = UnivariateLogisticRegression()

    times_train, events_train, _ = pickle.load(open('../Vectors_train.p', 'rb'))
    times_val, events_val, _ = pickle.load(open('../Vectors_val.p', 'rb'))
    times_test, events_test, _ = pickle.load(open('../Vectors_test.p', 'rb'))

    baseline.fit(np.array(times_train), np.array(events_train))
    print(baseline.evaluate(np.array(times_train), np.array(events_train)))
    print(baseline.evaluate(np.array(times_val), np.array(events_val)))
    print(baseline.evaluate(np.array(times_test), np.array(events_test)))
