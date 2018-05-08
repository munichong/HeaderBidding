import os, sys
import pandas as pd
from datetime import datetime, timedelta
from pymongo import MongoClient, ASCENDING, DESCENDING, HASHED
from AdExchangeImpressionsClass import AdExchangeImpressions


root = 'M:/Research Datasets/Header Bidding Data/'
client = MongoClient()
backfill_col = client['Header_Bidding']['NetworkBackfillImpressions']
matched_col = client['Header_Bidding']['BackfillAndAdxImpressions']  # NEW


def match(adx_series):
    # print(type(adx_series))
    query = {'URIs_pageno': adx_series['cleaned_url'],
             # ADX Time: Pacific Time; Backfill Time: Eastern Time
            'Time': {"$gte": adx_series['date_time'] + timedelta(hours=3) - timedelta(seconds=0),
                     "$lte": adx_series['date_time'] + timedelta(hours=3) + timedelta(seconds=0)},
            'AdPosition': adx_series['channels'],
            'Country': adx_series['expanded_country']}

    if adx_series['expanded_country'] == 'United States' and not pd.isnull(adx_series['expanded_state']):
        query['Region'] = adx_series['expanded_state']

    cand_backfill_imps = backfill_col.find(query)

    if cand_backfill_imps.count() == 0:
        return None

    # print(cand_backfill_imps.count(), 'impressions are retrieved')
    # print(adx_df['date_time'])
    for backfill_imp in cand_backfill_imps:
        backfill_imp.update(adx_series.to_dict())
        del backfill_imp['_id']
        return backfill_imp
    # print()


for datedir in sorted(os.listdir(os.path.join(root, 'ADX'))):
    if datedir[0] == '.':
        continue

    if datedir not in [
        # '2018.01.05',
        '2018.01.06'
    ]:
        continue

    for filename in sorted(os.listdir(os.path.join(root, 'ADX', datedir))):
        if filename[0] == '.':
            continue

        print('**************** DATE:', datedir, ':', filename)

        df = pd.read_csv(os.path.join(root, 'ADX', datedir, filename), delimiter=',')

        adx = AdExchangeImpressions(df)
        adx.preprocess()
        print("Finish preprocessing")

        output_buffer = []
        for i, row in adx.df.iterrows():
            matched_imp = match(row)
            if not matched_imp:
                continue
            output_buffer.append(matched_imp)

        if output_buffer:
            matched_col.insert(output_buffer)
        print(len(output_buffer), 'matched impressions are stored!')
