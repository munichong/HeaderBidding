import pickle
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from lifelines import KaplanMeierFitter


class UnivariateLogisticRegression:

    def __init__(self):
        self.lr = LogisticRegression(penalty='l2', C=1.0)

    def fit(self, X, y):
        return self.lr.fit(X, y, sample_weight=np.squeeze(X))

    def predict_proba(self, X):
        return self.lr.predict_proba(X)

    def predict(self, X):
        return self.lr.predict(X)

    def evaluate(self, X, y_bin_true, sample_weights=None):
        # print(np.array(X).shape, np.array(y_bin_true).shape)
        # print(self.lr.classes_)
        y_proba_pred = self.predict_proba(X)[:,1]
        # print(y_proba_pred)
        y_bin_pred = self.predict(X)
        # print(y_bin_pred)
        return log_loss(y_bin_true, y_proba_pred, sample_weight=sample_weights), \
               roc_auc_score(y_bin_true, y_proba_pred, sample_weight=sample_weights), \
               accuracy_score(y_bin_true, y_bin_pred, sample_weight=sample_weights)


class KaplanMeier:
    def __init__(self):
        self.kmf = KaplanMeierFitter()

    def fit(self, X, y):
        self.kmf.fit(durations=X, event_observed=y, left_censorship=True)
        return self

    def _get_one_prediction(self, x):
        return self.kmf.survival_function_.loc[self.kmf.survival_function_.ix[:,0]==x]

    def predict_proba(self, X):
        return pd.Series(map(self._get_one_prediction, X))

    def evalute(self, X, y_bin_true, sample_weights=None):
        pass





if __name__ == '__main__':
    def read_data(file_path):
        return pickle.load(open(file_path, 'rb'))

    def expand_dims(data, axis):
        return np.expand_dims(data, axis=axis)

    # baseline = UnivariateLogisticRegression()
    baseline = KaplanMeier()

    times_train, events_train = read_data('../Vectors_train.p')[:2]
    times_train = expand_dims(times_train, axis=1)
    times_val, events_val = read_data('../Vectors_val.p')[:2]
    times_val = expand_dims(times_val, axis=1)
    times_test, events_test = read_data('../Vectors_test.p')[:2]
    times_test = expand_dims(times_test, axis=1)

    baseline.fit(np.array(times_train), np.array(events_train))
    # print("Training Performance:\tlogloss=%.6f, auc=%.6f, accuracy=%.6f" %
    #       baseline.evaluate(times_train, np.array(events_train), sample_weights=np.squeeze(times_train)))
    # print("Validation Performance:\tlogloss=%.6f, auc=%.6f, accuracy=%.6f" %
    #       baseline.evaluate(times_val, np.array(events_val), sample_weights=np.squeeze(times_train)))
    # print("Test Performance:\tlogloss=%.6f, auc=%.6f, accuracy=%.6f" %
    #       baseline.evaluate(times_test, np.array(events_test), sample_weights=np.squeeze(times_train)))
