import csv, random, numpy as np, pickle
from scipy.sparse import coo_matrix
from failure_rate_prediction_conf.data_entry_class.ImpressionEntry import HEADER_BIDDING_KEYS

ADXWON_FEATVEC_IN_PATH, ADXLOSE_FEATVEC_IN_PATH = 'output/FeatVec_adxwon.csv', 'output/FeatVec_adxlose.csv'
ADXWON_hb_IN_PATH, ADXLOSE_hb_IN_PATH = 'output/HeaderBids_adxwon.csv', 'output/HeaderBids_adxlose.csv'

TRAIN_TMPOUT_STEMPATH, VAL_TMPOUT_STEMPATH, TEST_TMPOUT_STEMPATH = 'output/train_tmp', 'output/val_tmp', 'output/test_tmp'

TRAIN_OUT_PATH, VAL_OUT_PATH, TEST_OUT_PATH = 'output/TRAIN_SET.p', 'output/VAL_SET.p', 'output/TEST_SET.p'

TRAIN_PCT, VAL_PCT = 0.8, 0.1


def num_lines(file_path, exclude_header=False):
    raw_total = sum(1 for _ in csv.reader(open(file_path)))
    return raw_total - 1 if exclude_header else raw_total  # minus the header line

def print_lines_info(file_path, exclude_header=False):
    print("%d lines in the %s" % (num_lines(file_path, exclude_header=exclude_header), file_path))
    return num_lines(file_path)




def random_split(FEATVEC_PATH, hb_PATH):
    """
    This function assumes that the entire data can completely fit into the memory
    """
    global num_features
    with open(FEATVEC_PATH, newline='\n') as featvec_file, open(hb_PATH, newline='\n') as hb_file:
        num_features = int(next(featvec_file))  # skip the header
        next(hb_file)  # skip the header
        for featvec_line, hb_line in zip(featvec_file, hb_file):
            rand_float = random.random()  # Random float x, 0.0 <= x < 1.0
            if rand_float < TRAIN_PCT:
                training_featvec.append(featvec_line)
                training_hb.append(hb_line)
            elif TRAIN_PCT <= rand_float < TRAIN_PCT + VAL_PCT:
                validation_featvec.append(featvec_line)
                validation_hb.append(hb_line)
            else:
                test_featvec.append(featvec_line)
                test_hb.append(hb_line)




for infile_path in (ADXWON_FEATVEC_IN_PATH, ADXLOSE_FEATVEC_IN_PATH):
    print_lines_info(infile_path, exclude_header=True)
    
training_featvec, validation_featvec, test_featvec = [], [], []
training_hb, validation_hb, test_hb = [], [], []
num_features = 0


print("\nSplitting data...")
for featvec_path, headerbids_path in (
                    (ADXWON_FEATVEC_IN_PATH, ADXWON_hb_IN_PATH),
                    (ADXLOSE_FEATVEC_IN_PATH, ADXLOSE_hb_IN_PATH)):
    random_split(featvec_path, headerbids_path)

assert len(training_featvec) == len(training_hb)
assert len(validation_featvec) == len(validation_hb)
assert len(test_featvec) == len(test_hb)
#assert len(training_featvec) + len(validation_featvec) + len(test_featvec) == \
#       print_lines_info(ADXWON_FEATVEC_IN_PATH, exclude_header=True) + print_lines_info(ADXLOSE_FEATVEC_IN_PATH, exclude_header=True)



''' Write the split data sets (str; sparse format) to disk in order to easily check the partitions '''
print("\nWriting data...")
for featcsv_tmp, hb_tmp, tmpcsv_stempath in ((training_featvec, training_hb, TRAIN_TMPOUT_STEMPATH),
                            (validation_featvec, validation_hb, VAL_TMPOUT_STEMPATH),
                            (test_featvec, test_hb, TEST_TMPOUT_STEMPATH)):

    open(tmpcsv_stempath + '_featvec.csv', 'w', newline='\n').writelines(featcsv_tmp)
    open(tmpcsv_stempath + '_hb.csv', 'w', newline='\n').writelines(hb_tmp)
    assert print_lines_info(tmpcsv_stempath + '_featvec.csv') == print_lines_info(tmpcsv_stempath + '_hb.csv')








print("\nReading the temp csv files...")

def read_data(featvec_path, hb_path):
    times, events = [], []
    num_rows = print_lines_info(featvec_path)
    row_indices_fv, col_indices_fv, values_fv = [], [], []
    row_indices_hb, col_indices_hb, values_hb = [], [], []

    with open(featvec_path) as fvfile, open(hb_path) as hbfile:
        fv_reader = csv.reader(fvfile, delimiter=',')
        hb_reader = csv.reader(hbfile, delimiter=',')

        for i, (featvec, hb) in enumerate(zip(fv_reader, hb_reader)):
            times.append(float(featvec[0]))
            events.append(int(featvec[1]))
            try:
                for node in featvec[2:]:
                    col_index, val = node.split(':')
                    row_indices_fv.append(i)
                    col_indices_fv.append(int(col_index))
                    values_fv.append(float(val))

                for node in hb:
                    if node == '':
                        break
                    col_index, val = node.split(':')
                    row_indices_hb.append(i)
                    col_indices_hb.append(int(col_index))
                    values_hb.append(float(val))

            except MemoryError:
                print("MemoryError")
            print(i, num_rows)

    return np.array(times), \
         np.array(events), \
         coo_matrix((values_fv, (row_indices_fv, col_indices_fv)), shape=(num_rows, num_features)), \
         coo_matrix((values_hb, (row_indices_hb, col_indices_hb)), shape=(num_rows, len(HEADER_BIDDING_KEYS)))

for tmpcsv_stempath, pkl_path in ((TRAIN_TMPOUT_STEMPATH, TRAIN_OUT_PATH),
                            (VAL_TMPOUT_STEMPATH, VAL_OUT_PATH),
                            (TEST_TMPOUT_STEMPATH, TEST_OUT_PATH)):
    pickle.dump(read_data(tmpcsv_stempath + '_featvec.csv',
                          tmpcsv_stempath+ '_hb.csv'),
                open(pkl_path, 'wb'))
    print("DUMPED:", pkl_path)

