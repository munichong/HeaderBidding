import csv
import os
import pickle
import re
from collections import defaultdict, Counter
from pprint import pprint

from scipy import sparse
from scipy.sparse import coo_matrix

from failure_rate_prediction_journal.data_entry_class.ImpressionEntry import HEADER_BIDDING_KEYS, MIN_OCCURRENCE

PARTITION_DIR = '../output/missing_headerbids_data_partitions'
VECTOR_DIR = '../output/vectorization'


class Vectorizer:
    def __init__(self):
        self.counter = defaultdict(Counter)  # {Attribute1:Counter<features>, Attribute2:Counter<features>, ...}

    def fit(self, dir_path, file_filter_re):
        for filename in os.listdir(dir_path):
            if not re.match(file_filter_re, filename):
                # filename[ : len(agent_name)] != agent_name or filename[-len('train.p'):] != 'train.p':
                continue
            print("Fitting %s" % filename)
            for imp_entry in pickle.load(open(os.path.join(dir_path, filename), 'rb')):
                for k, v in imp_entry.entry.items():  # iterate all <fields:feature>
                    if type(v) == list:
                        self.counter[k].update(v)
                    elif type(v) == str:
                        self.counter[k][v] += 1
                    else:
                        self.counter[k][k] += 1  # for float or int features, occupy only one column

    def build_attr2idx(self):
        print('Building attr2idx')
        self.attr2idx = defaultdict(dict)  # {Attribute1: dict(feat1:i, ...), Attribute2: dict(feat1:i, ...), ...}
        self.num_features = 0
        for attr, feat_counter in self.counter.items():
            for feat in self.counter[attr]:
                ''' Skip rare features, leave it to the intercept '''
                if self.counter[attr][feat] < MIN_OCCURRENCE:
                    continue

                self.attr2idx[attr][feat] = self.num_features
                self.num_features += 1
        print('Finish building attr2idx')

    def transform_one_impression(self, imp_entry, agent_index):
        ''' Get prediction target '''
        header_bid = imp_entry.get_headerbids()[agent_index]
        if not imp_entry.is_qualified() or not header_bid:
            return None, None
        # return target + imp_entry.to_full_feature_vector(self.num_features)
        return header_bid, imp_entry.to_sparse_feature_vector(self.attr2idx)

    def transform(self, dir_path, agent_name, file_filter_re):
        agent_index = HEADER_BIDDING_KEYS.index(agent_name)
        header_bids = []
        feature_matrix = []
        # n = 0
        for filename in os.listdir(dir_path):
            if not re.match(file_filter_re, filename):
                continue
            print("Transforming %s" % filename)

            for imp_entry in pickle.load(open(os.path.join(dir_path, filename), 'rb')):
                header_bid, features = self.transform_one_impression(imp_entry, agent_index)
                if header_bid is not None:
                    header_bids.append([header_bid])
                    feature_matrix.append(features)

        return header_bids, feature_matrix


def output_one_agent_vector_files(vectorizer, output_dir, imp_files_path, agent_name):
    for dataset_type in ('train', 'val', 'test'):
        with open(os.path.join(output_dir,
                               '%s_featvec_%s.csv' % (agent_name, dataset_type)
                               ), 'a', newline='\n') as outfile_feat, \
                open(os.path.join(output_dir,
                                  '%s_headerbids_%s.csv' % (agent_name, dataset_type)
                                  ), 'a', newline='\n') as outfile_hb:
            writer_feat = csv.writer(outfile_feat, delimiter=',')
            writer_feat.writerow([vectorizer.num_features])  # the number of features
            writer_hb = csv.writer(outfile_hb, delimiter=',')
            hbs, mat = vectorizer.transform(imp_files_path, agent_name,
                                            r'%s_\d+_%s' % (agent_name, dataset_type))
            writer_feat.writerows(mat)
            writer_hb.writerows(hbs)


def build_vectors_across_all_agents():
    """
    Take all agents' features into account
    i.e., all agents share the same feature space.
    """
    vectorizer = Vectorizer()
    vectorizer.fit(PARTITION_DIR, r'.+_train\.p')
    vectorizer.build_attr2idx()
    print("\nCounter:")
    pprint(vectorizer.counter)
    print("\nAttr2Idx:")
    pprint(vectorizer.attr2idx)
    print("\n%d features\n" % vectorizer.num_features)

    ' delete old files '
    for file in os.listdir(VECTOR_DIR):
        os.remove(os.path.join(VECTOR_DIR, file))

    pickle.dump(vectorizer.counter, open(os.path.join(VECTOR_DIR, "counter.dict"), "wb"))
    pickle.dump(vectorizer.attr2idx, open(os.path.join(VECTOR_DIR, "attr2idx.dict"), "wb"))
    print("The counter and attr2idx are dumped")

    for agent_name in HEADER_BIDDING_KEYS:
        output_one_agent_vector_files(vectorizer,
                                      VECTOR_DIR,
                                      PARTITION_DIR,
                                      agent_name)


def featstr_to_sparsemat(dir_path):
    for filename in os.listdir(dir_path):
        if not re.match(r'.*_featvec_(train|val|test)\.csv', filename):
            continue
        print("Converting file %s" % filename)
        row_indices_fv, col_indices_fv, values_fv = [], [], []
        num_rows = 0
        row_gen = csv.reader(open(os.path.join(dir_path, filename)))
        num_features = int(next(row_gen)[0])
        for row in row_gen:
            for node in row:
                col_index, val = node.split(':')
                row_indices_fv.append(num_rows)
                col_indices_fv.append(int(col_index))
                values_fv.append(float(val))
            num_rows += 1
        sparse.save_npz(os.path.join(dir_path, filename[:-len('.csv')] + '.csr'),
                        coo_matrix((values_fv, (row_indices_fv, col_indices_fv)),
                                   shape=(num_rows, num_features)).tocsr())


if __name__ == "__main__":
    build_vectors_across_all_agents()
    featstr_to_sparsemat(VECTOR_DIR)
