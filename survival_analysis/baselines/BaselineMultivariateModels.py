import numpy as np, pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from survival_analysis.EvaluationMetrics import c_index
from scipy.sparse import hstack

class MultivariateSGDLogisticRegression:

    def __init__(self):
        self.lr = SGDClassifier(loss='log', penalty='l2', max_iter=10000, tol=1e-5,
                                # eta0=0.01, learning_rate='constant'
                                )

    def partial_fit(self, training_data, batch_size=10000):
        num_batches = 0
        for times_train, events_train, features_train in training_data.get_sparse_feat_vec_batch(batch_size):
            num_batches += 1
            print(num_batches, '/',  int(np.ceil(len(training_data.times) / batch_size)))
            times_features_train = hstack((np.expand_dims(times_train, axis=1), features_train))
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
        all_times = []
        weights = []
        for times, events, features in data.get_sparse_feat_vec_batch(10000):
            times_features = hstack((np.expand_dims(times, axis=1), features))
            y_bin_true.extend(events)
            y_proba_pred.extend(self.predict_proba(times_features)[:,1])
            y_bin_pred.extend(self.predict(times_features))
            weights.extend(times)
            all_times.extend(times)

        # Just for test, avoid slow c-index
        return log_loss(y_bin_true, y_proba_pred), 0.0, accuracy_score(y_bin_true, y_bin_pred)

        if not sample_weights:
            return log_loss(y_bin_true, y_proba_pred), \
                   c_index(y_bin_true, y_proba_pred, all_times), \
                   accuracy_score(y_bin_true, y_bin_pred)
        else:
            return log_loss(y_bin_true, y_proba_pred, sample_weight=weights), \
                   c_index(y_bin_true, y_proba_pred, all_times), \
                   accuracy_score(y_bin_true, y_bin_pred, sample_weight=weights)

