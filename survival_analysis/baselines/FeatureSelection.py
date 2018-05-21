import pickle, numpy as np
from sklearn.feature_selection import chi2
from survival_analysis.DataReader import SurvivalData

TRAIN_FILE_PATH = '../../Vectors_train.p'
VAL_FILE_PATH = '../../Vectors_val.p'
TEST_FILE_PATH = '../../Vectors_test.p'

def _read_data(file_path):
    return pickle.load(open(file_path, 'rb'))

def chi2_feature_selection(data):
    c = chi2(np.concatenate((data.times, data.sparse_features), axis=1), data.events)
    print(c.chi2)
    print(c.pval)







if __name__ == '__main__':
    training_data = SurvivalData(*_read_data(TRAIN_FILE_PATH))