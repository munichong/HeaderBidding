import os, sys
import pandas as pd
from datetime import datetime
# from pymongo import MongoClient
from NetworkBackfillImpressionsClass import NetworkBackfillImpressions


root = '/Volumes/New Volume/ad revenue data'

for datedir in sorted(os.listdir(os.path.join(root, 'NetworkBackfillImpressions'))):
    if datedir[0] == '.':
        continue

    print('DATE:', datedir)

    for filename in sorted(os.listdir(os.path.join(root, 'NetworkBackfillImpressions', datedir))):
        if filename[0] == '.':
            continue
        print(filename)

        df = pd.read_csv(os.path.join(root, 'NetworkBackfillImpressions', datedir, filename), header=0, delimiter='^')

        backfill = NetworkBackfillImpressions(df)
        backfill.preprocess()


    sys.exit(0)


