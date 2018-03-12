import os, sys
import pandas as pd
from datetime import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING, HASHED
from NetworkBackfillImpressionsClass import NetworkBackfillImpressions


root = 'M:/Research Datasets/Header Bidding Data/'
client = MongoClient()
backfill_col = client['Header_Bidding']['NetworkBackfillImpressions']
backfill_col.create_index([('URIs_pageno', ASCENDING),
                           ('Time', ASCENDING),
                           ('AdPosition', ASCENDING),
                           ('Country', ASCENDING),
                           ('Region', ASCENDING)])

start_process = False

for datedir in sorted(os.listdir(os.path.join(root, 'NetworkBackfillImpressions'))):
    if datedir[0] == '.':
        continue

    if datedir != '2018.01.05':
        continue

    for filename in sorted(os.listdir(os.path.join(root, 'NetworkBackfillImpressions', datedir))):
        if filename[0] == '.':
            continue

        # if filename == 'NetworkBackfillImpressions_330022_20180103_10':
        #     start_process = True
        #
        # if not start_process:
        #     continue

        print('**************** DATE:', datedir, ':', filename)

        df = pd.read_csv(os.path.join(root, 'NetworkBackfillImpressions', datedir, filename), header=0, delimiter='^')

        backfill = NetworkBackfillImpressions(df)
        backfill.preprocess()
        backfill_col.insert_many(backfill.df.to_dict('r'))

        print(len(backfill.df), "STORED!")


    sys.exit(0)


