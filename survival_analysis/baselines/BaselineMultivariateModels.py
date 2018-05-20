import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score


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