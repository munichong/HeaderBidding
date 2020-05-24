import os
import sys

import pandas as pd
from NetworkBackfillImpressionsClass import NetworkBackfillImpressions
from NetworkImpressionsClass import NetworkImpressions
from pymongo import MongoClient, ASCENDING

root = 'M:/Research Datasets/Header Bidding Data/'
client = MongoClient()


def import_dataset(dataname, ImpressionClass):
    col = client['Header_Bidding'][dataname]
    col.create_index([('URIs_pageno', ASCENDING),
                      ('Time', ASCENDING),
                      ('AdPosition', ASCENDING),
                      ('Country', ASCENDING),
                      ('Region', ASCENDING)])

    start_process = False

    for datedir in sorted(os.listdir(os.path.join(root, dataname))):
        if datedir[0] == '.':
            continue

        if datedir != '2018.04.15':
            continue

        for filename in sorted(os.listdir(os.path.join(root, dataname, datedir))):
            if filename[0] == '.':
                continue

            # if filename == 'NetworkBackfillImpressions_330022_20180415_19':
            #     start_process = True
            #
            # if not start_process:
            #     continue

            print('**************** DATE:', datedir, ':', filename)

            df = pd.read_csv(os.path.join(root, dataname, datedir, filename), header=0, delimiter='^')

            imp_inst = ImpressionClass(df)
            imp_inst.preprocess()
            col.insert_many(imp_inst.df.to_dict('r'))

            print(len(imp_inst.df), "STORED!")

        sys.exit(0)


import_dataset('NetworkBackfillImpressions', NetworkBackfillImpressions)
import_dataset('NetworkImpressions', NetworkImpressions)
