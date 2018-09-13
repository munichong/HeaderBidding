import csv, random, numpy as np, pickle
from scipy.sparse import coo_matrix
from survival_analysis.data_entry_class.ImpressionEntry import HEADER_BIDDING_KEYS

ADXWON_FEATVEC_IN_PATH, ADXLOSE_FEATVEC_IN_PATH = '../FeatVec_adxwon.csv', '../FeatVec_adxlose.csv'
ADXWON_HD_IN_PATH, ADXLOSE_HD_IN_PATH = '../HeaderBids_adxwon.csv', '../HeaderBids_adxlose.csv'

TRAIN_TMPOUT_STEMPATH, VAL_TMPOUT_STEMPATH, TEST_TMPOUT_STEMPATH = '../train_tmp', '../val_tmp', '../test_tmp'

TRAIN_OUT_PATH, VAL_OUT_PATH, TEST_OUT_PATH = '../TRAIN_SET.p', '../VAL_SET.p', '../TEST_SET.p'

TRAIN_PCT, VAL_PCT = 0.8, 0.1


def num_lines(file_path, exclude_header=False):
    raw_total = sum(1 for _ in csv.reader(open(file_path)))
    return raw_total - 1 if exclude_header else raw_total  # minus the header line

def print_lines_info(file_path, exclude_header=False):
    print("%d lines in the %s" % (num_lines(file_path, exclude_header=exclude_header), file_path))
    return num_lines(file_path)




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
                test_hd.append(hd_line)




for infile_path in (ADXWON_FEATVEC_IN_PATH, ADXLOSE_FEATVEC_IN_PATH):
    print_lines_info(infile_path, exclude_header=True)
    
training_featvec, validation_featvec, test_featvec = [], [], []
training_hd, validation_hd, test_hd = [], [], []
num_features = 0


print("\nSplitting data...")
for featvec_path, headerbids_path in (
                    (ADXWON_FEATVEC_IN_PATH, ADXWON_HD_IN_PATH),
                    (ADXLOSE_FEATVEC_IN_PATH, ADXLOSE_HD_IN_PATH)):
    random_split(featvec_path, headerbids_path)

assert len(training_featvec) == len(training_hd)
assert len(validation_featvec) == len(validation_hd)
assert len(test_featvec) == len(test_hd)
#assert len(training_featvec) + len(validation_featvec) + len(test_featvec) == \
#       print_lines_info(ADXWON_FEATVEC_IN_PATH, exclude_header=True) + print_lines_info(ADXLOSE_FEATVEC_IN_PATH, exclude_header=True)



''' Write the split data sets (str; sparse format) to disk in order to easily check the partitions '''
print("\nWriting data...")
for featcsv_tmp, hd_tmp, tmpcsv_stempath in ((training_featvec, training_hd, TRAIN_TMPOUT_STEMPATH),
                            (validation_featvec, validation_hd, VAL_TMPOUT_STEMPATH),
                            (test_featvec, test_hd, TEST_TMPOUT_STEMPATH)):

    open(tmpcsv_stempath + '_featvec.csv', 'w', newline='\n').writelines(featcsv_tmp)
    open(tmpcsv_stempath + '_hd.csv', 'w', newline='\n').writelines(hd_tmp)
    assert print_lines_info(tmpcsv_stempath + '_featvec.csv') == print_lines_info(tmpcsv_stempath + '_hd.csv')








print("\nReading the temp csv files...")

def read_data(featvec_path, hd_path):
    times, events = [], []
    num_rows = print_lines_info(featvec_path)
    row_indices_fv, col_indices_fv, values_fv = [], [], []
    row_indices_hd, col_indices_hd, values_hd = [], [], []

    with open(featvec_path) as fvfile, open(hd_path) as hdfile:
        fv_reader = csv.reader(fvfile, delimiter=',')
        hd_reader = csv.reader(hdfile, delimiter=',')

        for i, (featvec, hd) in enumerate(zip(fv_reader, hd_reader)):
            times.append(float(featvec[0]))
            events.append(int(featvec[1]))
            try:
                for node in featvec[2:]:
                    col_index, val = node.split(':')
                    row_indices_fv.append(i)
                    col_indices_fv.append(int(col_index))
                    values_fv.append(float(val))

                for node in hd:
                    if node == '':
                        break
                    col_index, val = node.split(':')
                    row_indices_hd.append(i)
                    col_indices_hd.append(int(col_index))
                    values_hd.append(float(val))

            except MemoryError:
                print("MemoryError")
            print(i, num_rows)

    return np.array(times), \
         np.array(events), \
         coo_matrix((values_fv, (row_indices_fv, col_indices_fv)), shape=(num_rows, num_features)), \
         coo_matrix((values_hd, (row_indices_hd, col_indices_hd)), shape=(num_rows, len(HEADER_BIDDING_KEYS)))

for tmpcsv_stempath, pkl_path in ((TRAIN_TMPOUT_STEMPATH, TRAIN_OUT_PATH),
                            (VAL_TMPOUT_STEMPATH, VAL_OUT_PATH),
                            (TEST_TMPOUT_STEMPATH, TEST_OUT_PATH)):
    pickle.dump(read_data(tmpcsv_stempath + '_featvec.csv',
                          tmpcsv_stempath+ '_hd.csv'),
                open(pkl_path, 'wb'))
    print("DUMPED:", pkl_path)

