import os, re, csv, pickle
import pandas as pd
from pprint import pprint
from failure_rate_prediction_conf.data_entry_class.NetworkBackfillImpressionEntry import NetworkBackfillImpressionEntry
from failure_rate_prediction_conf.data_entry_class.NetworkImpressionEntry import NetworkImpressionEntry
from failure_rate_prediction_conf.data_entry_class.ImpressionEntry import HEADER_BIDDING_KEYS, MIN_OCCURRENCE
from collections import defaultdict, Counter


PARTITION_DIR = '../output/missing_headerbids_data_partitions'
VECTOR_ONE_DIR = '../output/one_agent_vectorization'
VECTOR_ALL_DIR = '../output/all_agents_vectorization'


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

    # def fit_one_agent(self, dir_path, agent_name):
    #     """
    #     Build dictionary for all hd agents so that they share the same feature space
    #     """
    #     for filename in os.listdir(dir_path):
    #         if filename[ : len(agent_name)] != agent_name or filename[-len('train.p'):] != 'train.p':
    #             continue
    #         print("Fitting %s" % filename)
    #         for imp_entry in pickle.load(open(os.path.join(dir_path, filename), 'rb')):
    #             for k, v in imp_entry.entry.items():  # iterate all <fields:feature>
    #                 if type(v) == list:
    #                     self.counter[k].update(v)
    #                 elif type(v) == str:
    #                     self.counter[k][v] += 1
    #                 else:
    #                     self.counter[k][k] += 1  # for float or int features, occupy only one column
    #
    #
    # def fit_all_agents(self, dir_path):
    #     for agent_name in HEADER_BIDDING_KEYS:
    #         self.fit_one_agent(dir_path, agent_name)


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
        n = 0
        for filename in os.listdir(dir_path):
            if not re.match(file_filter_re, filename):
                continue
            print("Transforming %s" % filename)
            if n % 1000000 == 0:
                yield header_bids, feature_matrix
                header_bids.clear()
                feature_matrix.clear()
            n += 1

            for imp_entry in pickle.load(open(os.path.join(dir_path, filename), 'rb')):
                header_bid, features = self.transform_one_impression(imp_entry, agent_index)
                if header_bid is not None:
                    header_bids.append([header_bid])
                    feature_matrix.append(features)

        yield header_bids, feature_matrix
        header_bids.clear()
        feature_matrix.clear()

    # def transform_all_agents(self, dir_path):
    #     for agent_name in HEADER_BIDDING_KEYS:
    #         yield self.transform_one_agent(dir_path, agent_name)



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
            for hbs, mat in vectorizer.transform(imp_files_path, agent_name,
                                                r'%s_\d+_%s' % (agent_name, dataset_type)):
                writer_feat.writerows(mat)
                writer_hb.writerows(hbs)
                pickle.dump(mat, open(os.path.join(output_dir,
                               '%s_featvec_%s.p' % (agent_name, dataset_type)
                               ), "wb"))
                pickle.dump(hbs, open(os.path.join(output_dir,
                                '%s_headerbids_%s.p' % (agent_name, dataset_type)
                                ), "wb"))


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
    for file in os.listdir(VECTOR_ALL_DIR):
        os.remove(os.path.join(VECTOR_ALL_DIR, file))

    # counter does NOT contain header bidding.
    # counter contains the most common feature in each attribute
    pickle.dump(vectorizer.counter, open(os.path.join(VECTOR_ALL_DIR, "counter.dict"), "wb"))
    # attr2idx does NOT contain header bidding.
    # attr2idx does NOT contain the most common feature in each attribute
    pickle.dump(vectorizer.attr2idx, open(os.path.join(VECTOR_ALL_DIR, "attr2idx.dict"), "wb"))
    print("The counter and attr2idx are dumped")

    for agent_name in HEADER_BIDDING_KEYS:
        output_one_agent_vector_files(vectorizer,
                                      VECTOR_ALL_DIR,
                                      PARTITION_DIR,
                                      agent_name)

def build_vectors_for_one_agent():
    """
    Only take one agent's features into account
    i.e., an agents has a unique and narrow feature space.
    """
    ' delete old files '
    for file in os.listdir(VECTOR_ONE_DIR):
        os.remove(os.path.join(VECTOR_ONE_DIR, file))

    for agent_name in HEADER_BIDDING_KEYS:
        print("\n================================================")
        print("Processing agent %s ..." % agent_name)
        vectorizer = Vectorizer()
        vectorizer.fit(PARTITION_DIR, r'%s_\d+_train\.p' % agent_name)
        vectorizer.build_attr2idx()
        print("\nCounter:")
        pprint(vectorizer.counter)
        print("\nAttr2Idx:")
        pprint(vectorizer.attr2idx)
        print("\n%d features\n" % vectorizer.num_features)

        # counter does NOT contain header bidding.
        # counter contains the most common feature in each attribute
        pickle.dump(vectorizer.counter, open(os.path.join(VECTOR_ONE_DIR, agent_name + '_' + "counter.dict"), "wb"))
        # attr2idx does NOT contain header bidding.
        # attr2idx does NOT contain the most common feature in each attribute
        pickle.dump(vectorizer.attr2idx, open(os.path.join(VECTOR_ONE_DIR, agent_name + '_' + "attr2idx.dict"), "wb"))

        output_one_agent_vector_files(vectorizer,
                                      VECTOR_ONE_DIR,
                                      PARTITION_DIR,
                                      agent_name)
        print("Finish agent %s" % agent_name)



if __name__ == "__main__":
    build_vectors_across_all_agents()
    build_vectors_for_one_agent()