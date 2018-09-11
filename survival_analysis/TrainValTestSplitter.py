import csv, random, numpy as np, pickle
from random import shuffle
from scipy.sparse import coo_matrix

ADXWON_FEATVEC_IN_PATH, ADXLOSE_FEATVEC_IN_PATH = '../FeatVec_adxwon.csv', '../FeatVec_adxlose.csv'
ADXWON_HEADERBIDS_IN_PATH, ADXLOSE_HEADERBIDS_IN_PATH = '../HeaderBids_adxwon.csv', '../HeaderBids_adxlose.csv'

TRAIN_OUTCSV_PATH, VAL_OUTCSV_PATH, TEST_OUTCSV_PATH = '../Vectors_train.csv', '../Vectors_val.csv', '../Vectors_test.csv'
TRAIN_OUTPKL_PATH, VAL_OUTPKL_PATH, TEST_OUTPKL_PATH = '../Vectors_train.p', '../Vectors_val.p', '../Vectors_test.p'

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
                training_data.append((featvec_line, hd_line))
            elif TRAIN_PCT <= rand_float < TRAIN_PCT + VAL_PCT:
                validation_data.append((featvec_line, hd_line))
            else:
                test_data.append((featvec_line, hd_line))

training_data, validation_data, test_data = [], [], [] # [(features_sparse_str, headerbids_sparse_str), ...]
num_features = 0

print()
print("Splitting data...")
for featvec_path, headerbids_path in ((ADXWON_FEATVEC_IN_PATH, ADXWON_HEADERBIDS_IN_PATH),
                    (ADXLOSE_FEATVEC_IN_PATH, ADXLOSE_HEADERBIDS_IN_PATH)):
    random_split(featvec_path, headerbids_path)




print()
print("Writing data...")
for data, outfile_path in ((training_data, TRAIN_OUTCSV_PATH),
                            (validation_data, VAL_OUTCSV_PATH),
                            (test_data, TEST_OUTCSV_PATH)):
    shuffle(data)
    open(outfile_path, 'w', newline='\n').writelines(data)
    print_lines_info(outfile_path)



def read_data(file_path):
    times, events = [], []
    row_indices, col_indices, values = [], [], []
    num_rows = 0
    with open(file_path) as infile:
        csv_reader = csv.reader(infile, delimiter=',')
        for row_index, row in enumerate(csv_reader):
            num_rows += 1
            times.append(float(row[0]))
            events.append(int(row[1]))
            for node in row[2:]:
                col_index, val = node.split(':')
                row_indices.append(row_index)
                col_indices.append(int(col_index))
                values.append(float(val))
    return np.array(times), \
           np.array(events), \
           coo_matrix((values, (row_indices, col_indices)), shape=(num_rows, num_features))

for csvfile, picklefile in ((TRAIN_OUTCSV_PATH, TRAIN_OUTPKL_PATH),
                            (VAL_OUTCSV_PATH, VAL_OUTPKL_PATH),
                            (TEST_OUTCSV_PATH, TEST_OUTPKL_PATH)):
    pickle.dump(read_data(csvfile), open(picklefile, 'wb'))
    print("DUMP:", picklefile)