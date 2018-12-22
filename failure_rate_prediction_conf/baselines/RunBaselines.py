import pickle, numpy as np

from failure_rate_prediction_conf.DataReader import SurvivalData
from failure_rate_prediction_conf.baselines.BaselineUnivariateModels import UnivariateLogisticRegression, KaplanMeier
from failure_rate_prediction_conf.baselines.BaselineMultivariateModels import MultivariateSGDLogisticRegression


TRAIN_FILE_PATH = '../output/TRAIN_SET.p'
VAL_FILE_PATH = '../output/VAL_SET.p'
TEST_FILE_PATH = '../output/TEST_SET.p'

def _read_data(file_path):
    return pickle.load(open(file_path, 'rb'))

def _expand_dims(data, axis=1):
    return np.expand_dims(data, axis=axis)

def run_univariate_baselines(Baseline):
    baseline = Baseline()

    times_train, events_train = _read_data(TRAIN_FILE_PATH)[:2]
    times_train = _expand_dims(times_train, axis=1)
    times_val, events_val = _read_data(VAL_FILE_PATH)[:2]
    times_val = _expand_dims(times_val, axis=1)
    times_test, events_test = _read_data(TEST_FILE_PATH)[:2]
    times_test = _expand_dims(times_test, axis=1)

    baseline.fit(np.array(times_train), np.array(events_train))
    print("Training Performance:\tlogloss=%.6f, c-index=%.6f, accuracy=%.6f" %
          baseline.evaluate(times_train, np.array(events_train), sample_weights=np.squeeze(times_train)))
    print("Validation Performance:\tlogloss=%.6f, c-index=%.6f, accuracy=%.6f" %
          baseline.evaluate(times_val, np.array(events_val), sample_weights=np.squeeze(times_val)))
    print("Test Performance:\tlogloss=%.6f, c-index=%.6f, accuracy=%.6f" %
          baseline.evaluate(times_test, np.array(events_test), sample_weights=np.squeeze(times_test)))



def run_multivariate_baselines(Baseline, sample_weights=None):

    baseline = Baseline()
    training_data = SurvivalData(*_read_data(TRAIN_FILE_PATH))
    baseline.partial_fit(training_data)

    train_data = SurvivalData(*_read_data(TRAIN_FILE_PATH))
    print("Training Performance:\tlogloss=%.6f, c-index=%.6f, accuracy=%.6f" %
          baseline.evaluate(train_data, sample_weights=sample_weights))

    val_data = SurvivalData(*_read_data(VAL_FILE_PATH))
    print("Validation Performance:\tlogloss=%.6f, c-index=%.6f, accuracy=%.6f" %
          baseline.evaluate(val_data, sample_weights=sample_weights))

    test_data = SurvivalData(*_read_data(TEST_FILE_PATH))
    print("Test Performance:\tlogloss=%.6f, c-index=%.6f, accuracy=%.6f" %
          baseline.evaluate(test_data, sample_weights=sample_weights))




if __name__ == '__main__':
    # run_univariate_baselines(UnivariateLogisticRegression)
    # run_univariate_baselines(KaplanMeier)
    run_multivariate_baselines(MultivariateSGDLogisticRegression, sample_weights='time')