import os, sys
import pandas as pd
from datetime import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING, HASHED
from NetworkImpressionsClass import NetworkImpressions


root = 'M:/Research Datasets/Header Bidding Data/'
client = MongoClient()
headerbidding_col = client['Header_Bidding']['NetworkImpressions']
# headerbidding_col.create_index([('URIs_pageno', ASCENDING),
#                            ('Time', ASCENDING),
#                            ('AdPosition', ASCENDING),
#                            ('Country', ASCENDING),
#                            ('Region', ASCENDING)])


for datedir in sorted(os.listdir(os.path.join(root, 'NetworkImpressions'))):
    if datedir[0] == '.':
        continue

    if datedir != '2018.01.05':
        continue

    for filename in sorted(os.listdir(os.path.join(root, 'NetworkImpressions', datedir))):
        if filename[0] == '.':
            continue

        print('**************** DATE:', datedir, ':', filename)

        df = pd.read_csv(os.path.join(root, 'NetworkImpressions', datedir, filename), header=0, delimiter='^')

        headerbidding = NetworkImpressions(df)
        headerbidding.preprocess()
        headerbidding_col.insert_many(headerbidding.df.to_dict('r'))

        print(len(headerbidding.df), "STORED!")


    sys.exit(0)


