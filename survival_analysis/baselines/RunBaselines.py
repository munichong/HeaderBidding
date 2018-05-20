import pickle, numpy as np

from survival_analysis.DataReader import SurvivalData
from survival_analysis.baselines.BaselineUnivariateModels import UnivariateLogisticRegression, KaplanMeier


def run_univariate_baselines(Baseline):
    def read_data(file_path):
        return pickle.load(open(file_path, 'rb'))

    def expand_dims(data, axis):
        return np.expand_dims(data, axis=axis)

    baseline = Baseline()

    times_train, events_train = read_data('../Vectors_train.p')[:2]
    times_train = expand_dims(times_train, axis=1)
    times_val, events_val = read_data('../Vectors_val.p')[:2]
    times_val = expand_dims(times_val, axis=1)
    times_test, events_test = read_data('../Vectors_test.p')[:2]
    times_test = expand_dims(times_test, axis=1)

    baseline.fit(np.array(times_train), np.array(events_train))
    print("Training Performance:\tlogloss=%.6f, auc=%.6f, accuracy=%.6f" %
          baseline.evaluate(times_train, np.array(events_train), sample_weights=np.squeeze(times_train)))
    print("Validation Performance:\tlogloss=%.6f, auc=%.6f, accuracy=%.6f" %
          baseline.evaluate(times_val, np.array(events_val), sample_weights=np.squeeze(times_train)))
    print("Test Performance:\tlogloss=%.6f, auc=%.6f, accuracy=%.6f" %
          baseline.evaluate(times_test, np.array(events_test), sample_weights=np.squeeze(times_train)))



def run_multivariate_baselines(Baseline):
    pass



if __name__ == '__main__':
    run_univariate_baselines(UnivariateLogisticRegression)
    # run_univariate_baselines(KaplanMeier)