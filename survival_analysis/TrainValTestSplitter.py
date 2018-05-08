import csv, random

ADXWON_INFILE_PATH, ADXLOSE_INFILE_PATH = '../Vectors_adxwon.csv', '../Vectors_adxlose.csv'
TRAIN_OUTFILE_PATH, VAL_OUTFILE_PATH, TEST_OUTFILE_PATH = '../Vectors_train.csv', '../Vectors_val.csv', '../Vectors_test.csv'


TRAIN_PCT, VAL_PCT = 0.8, 0.1


def num_lines(file_path):
    return sum(1 for _ in csv.reader(open(file_path))) - 1  # minus the header line

def print_lines_info(file_path):
    print("%d lines in the %s" % (num_lines(file_path), file_path))

print_lines_info(ADXWON_INFILE_PATH)
print_lines_info(ADXLOSE_INFILE_PATH)


writer_train = open(TRAIN_OUTFILE_PATH, 'w', newline='\n')
writer_val = open(VAL_OUTFILE_PATH, 'w', newline='\n')
writer_test = open(TEST_OUTFILE_PATH, 'w', newline='\n')

def split_to_files(INFILE_PATH):
    with open(INFILE_PATH, newline='\n') as infile:
        next(infile)  # skip the header
        for line in infile:
            rand_float = random.random()  # Random float x, 0.0 <= x < 1.0
            if rand_float < TRAIN_PCT:
                writer_train.write(line)
            elif TRAIN_PCT <= rand_float < TRAIN_PCT + VAL_PCT:
                writer_val.write(line)
            else:
                writer_test.write(line)

split_to_files(ADXWON_INFILE_PATH)
split_to_files(ADXLOSE_INFILE_PATH)

print_lines_info(TRAIN_OUTFILE_PATH)
print_lines_info(VAL_OUTFILE_PATH)
print_lines_info(TEST_OUTFILE_PATH)


