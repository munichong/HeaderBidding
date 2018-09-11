import csv, random, numpy as np, pickle
from scipy.sparse import coo_matrix
from survival_analysis.data_entry_class.ImpressionEntry import HEADER_BIDDING_KEYS

ADXWON_FEATVEC_IN_PATH, ADXLOSE_FEATVEC_IN_PATH = '../FeatVec_adxwon.csv', '../FeatVec_adxlose.csv'
ADXWON_HD_IN_PATH, ADXLOSE_HD_IN_PATH = '../HeaderBids_adxwon.csv', '../HeaderBids_adxlose.csv'

TRAIN_OUT_PATH, VAL_OUT_PATH, TEST_OUT_PATH = '../TRAIN_SET.p', '../VAL_SET.p', '../TEST_SET.p'

TRAIN_PCT, VAL_PCT = 0.8, 0.1


def num_lines(file_path):
    return sum(1 for _ in csv.reader(open(file_path))) - 1  # minus the header line

def print_lines_info(file_path):
    print("%d lines in the %s" % (num_lines(file_path), file_path))

for infile_path in (ADXWON_FEATVEC_IN_PATH, ADXLOSE_FEATVEC_IN_PATH):
    print_lines_info(infile_path)




def random_split(FEATVEC_PATH, HD_PATH):
    """
    This function assumes that the entire data can completely fit into the memory
    """
    global num_features
    with open(FEATVEC_PATH, newline='\n') as featvec_file, open(HD_PATH, newline='\n') as hd_file:
        num_features = int(next(featvec_file))  # skip the header
        next(hd_file)  # skip the header
        for featvec_line, hd_line in zip(featvec_file, hd_file):
            rand_float = random.random()  # Random float x, 0.0 <= x < 1.0
            if rand_float < TRAIN_PCT:
                training_featvec.append(featvec_line)
                training_hd.append(hd_line)
            elif TRAIN_PCT <= rand_float < TRAIN_PCT + VAL_PCT:
                validation_featvec.append(featvec_line)
                validation_hd.append(hd_line)
            else:
                test_featvec.append(featvec_line)
                test_hd.append(test_hd)

training_featvec, validation_featvec, test_featvec = [], [], []
training_hd, validation_hd, test_hd = [], [], []
num_features = 0

print()
print("Splitting data...")
for featvec_path, headerbids_path in (
                    (ADXWON_FEATVEC_IN_PATH, ADXWON_HD_IN_PATH),
                    (ADXLOSE_FEATVEC_IN_PATH, ADXLOSE_HD_IN_PATH)):
    random_split(featvec_path, headerbids_path)




for featvec_set, hd_set, out_path in ((training_featvec, training_hd, TRAIN_OUT_PATH),
                            (validation_featvec, validation_hd, VAL_OUT_PATH),
                            (test_featvec, test_hd, TEST_OUT_PATH)):
    times, events = [], []
    row_indices_fv, col_indices_fv, values_fv = [], [], []
    row_indices_hd, col_indices_hd, values_hd = [], [], []
    num_rows = 0
    for i, (featvec, hd) in enumerate(zip(featvec_set, hd_set)):
        num_rows += 1
        times.append(float(featvec[0]))
        events.append(int(featvec[1]))
        for node in featvec[2:]:
            col_index, val = node.split(':')
            row_indices_fv.append(i)
            col_indices_fv.append(int(col_index))
            values_fv.append(float(val))

        for node in hd:
            col_index, val = node.split(':')
            row_indices_hd.append(i)
            col_indices_hd.append(int(col_index))
            values_hd.append(float(val))

    pickle.dump((np.array(times),
                 np.array(events),
                 coo_matrix((values_fv, (row_indices_fv, col_indices_fv)), shape=(num_rows, num_features)),
                 coo_matrix((values_hd, (row_indices_hd, col_indices_hd)), shape=(num_rows, len(HEADER_BIDDING_KEYS))),
                 ),
                open(out_path, 'wb'))
    print("DUMP:", out_path)

