import pickle, numpy as np
from pprint import pprint
from sklearn.feature_selection import chi2
from survival_analysis.DataReader import SurvivalData
from tabulate import tabulate

TRAIN_FILE_PATH = '../../Vectors_train.p'
VAL_FILE_PATH = '../../Vectors_val.p'
TEST_FILE_PATH = '../../Vectors_test.p'

def _read_data(file_path):
    return pickle.load(open(file_path, 'rb'))

def chi2_feature_selection(X, y, attr2idx=None):
    chi, pval = chi2(X, y)
    output = []
    if not attr2idx:
        print(sorted(enumerate(zip(chi, pval)), key=lambda x:x[1], reverse=True))
    else:
        idx2attr = {idx: ':'.join([attr, feat]) for attr, idx_dict in attr2idx.items() for feat, idx in idx_dict.items()}
        for index, (chi, p) in sorted(enumerate(zip(chi, pval)), key=lambda x: x[1], reverse=True):
            if index not in idx2attr:
                print("Index %d is not in the dictionary" % index)
                output.append(('<Header Bid>', index, chi, p))
                continue
            output.append((idx2attr[index], index, chi, p))
            # print("%s(%d)\tchi2 = %.4f, pval = %.4f" % (idx2attr[index], index, chi, p))
            print(tabulate(output,
                            headers=['Field:Feature', 'index', 'chi2', 'pval'],
                            tablefmt='orgtbl'))



if __name__ == '__main__':
    training_data = SurvivalData(*_read_data(TRAIN_FILE_PATH))
    attr2idx = pickle.load(open('../../attr2idx.dict', 'rb'))
    chi2_feature_selection(training_data.sparse_features, training_data.events, attr2idx=attr2idx)
    chi2_feature_selection(np.expand_dims(training_data.times, -1), training_data.events)