import numpy as np, pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score


class MultivariateSGDLogisticRegression:

    def __init__(self):
        self.lr = SGDClassifier(loss='log', penalty='none', max_iter=1000, tol=1e-3)

    def partial_fit(self, training_data):
        for times_train, events_train, features_train in training_data.make_batch(10000):
            times_features_train = np.concatenate((np.expand_dims(times_train, axis=1), features_train), axis=1)
            self.lr.partial_fit(times_features_train, events_train, classes=[0, 1])
        return self

    def predict_proba(self, X):
        return self.lr.predict_proba(X)

    def predict(self, X):
        return self.lr.predict(X)

    def evaluate(self, data, sample_weights=None):
        y_proba_pred = []
        y_bin_pred = []
        y_bin_true = []
        weights = []
        for times, events, features in data.make_batch(10000):
            times_features = np.concatenate((np.expand_dims(times, axis=1), features), axis=1)
            y_bin_true.extend(events)
            y_proba_pred.extend(self.predict_proba(times_features)[:,1])
            y_bin_pred.extend(self.predict(times_features))
            weights.extend(times)

        if not sample_weights:
            return log_loss(y_bin_true, y_proba_pred), \
                   roc_auc_score(y_bin_true, y_proba_pred), \
                   accuracy_score(y_bin_true, y_bin_pred)
        else:
            return log_loss(y_bin_true, y_proba_pred, sample_weight=weights), \
                   roc_auc_score(y_bin_true, y_proba_pred, sample_weight=weights), \
                   accuracy_score(y_bin_true, y_bin_pred, sample_weight=weights)

