import csv, random
from random import shuffle

ADXWON_INFILE_PATH, ADXLOSE_INFILE_PATH = '../Vectors_adxwon.csv', '../Vectors_adxlose.csv'
TRAIN_OUTFILE_PATH, VAL_OUTFILE_PATH, TEST_OUTFILE_PATH = '../Vectors_train.csv', '../Vectors_val.csv', '../Vectors_test.csv'

TRAIN_PCT, VAL_PCT = 0.8, 0.1


def num_lines(file_path):
    return sum(1 for _ in csv.reader(open(file_path))) - 1  # minus the header line

def print_lines_info(file_path):
    print("%d lines in the %s" % (num_lines(file_path), file_path))

print_lines_info(ADXWON_INFILE_PATH)
print_lines_info(ADXLOSE_INFILE_PATH)


training_data, validation_data, test_data = [], [], []

def random_split(INFILE_PATH):
    """
    This function assumes that the entire data can completely fit into the memory
    """
    with open(INFILE_PATH, newline='\n') as infile:
        next(infile)  # skip the header
        for line in infile:
            rand_float = random.random()  # Random float x, 0.0 <= x < 1.0
            if rand_float < TRAIN_PCT:
                training_data.append(line)
            elif TRAIN_PCT <= rand_float < TRAIN_PCT + VAL_PCT:
                validation_data.append(line)
            else:
                test_data.append(line)


for infile_path in (ADXWON_INFILE_PATH, ADXLOSE_INFILE_PATH):
    random_split(infile_path)

for data, outfile_path in ((training_data, TRAIN_OUTFILE_PATH),
                            (validation_data, VAL_OUTFILE_PATH),
                            (test_data, TEST_OUTFILE_PATH)):
    shuffle(data)
    open(outfile_path, 'w', newline='\n').writelines(data)

