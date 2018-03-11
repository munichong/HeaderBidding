import json, re
import pandas as pd
import numpy as np
from datetime import datetime
from urllib.request import urlopen


HEADER_BIDDING_KEYS = ('mnetbidprice',
                       'mnet_abd',
                       'mnet_fbcpm',
                       'amznbid',
                       'fb_bid_price_cents')

FORBES_API_ROOT = 'https://forbesapis.forbes.com/forbesapi/content/all.json/?code=a6016ad7796e2165bba73787d68f3162b29f9bd7'

class NetworkBackfillImpressions():
    def __init__(self, file_content):
        self.df = file_content


    def remove_columns(self):
        '''
       ['Time', 'UserId', 'IP', 'AdvertiserId', 'OrderId', 'LineItemId',
       'CreativeId', 'CreativeVersion', 'CreativeSize', 'AdUnitId',
       'CustomTargeting', 'Domain', 'CountryId', 'Country', 'RegionId',
       'Region', 'MetroId', 'Metro', 'CityId', 'City', 'PostalCodeId',
       'PostalCode', 'BrowserId', 'Browser', 'OSId', 'OS', 'BandWidth',
       'BandwidthId', 'TimeUsec', 'Product', 'ActiveViewEligibleImpression',
       'DeviceCategory', 'GfpContentId', 'KeyPart', 'PodPosition',
       'PublisherProvidedID', 'RequestedAdUnitSizes', 'TargetedCustomCriteria',
       'TimeUsec2', 'VideoPosition', 'VideoFallbackPosition']

        print(df.columns)
        '''
        self.df = self.df.drop(columns=['IP', 'CreativeVersion', 'Domain',
                                        'CountryId', 'RegionId', 'MetroId', 'CityId', 'PostalCodeId',
                                        'BrowserId', 'OSId', 'BandwidthId', 'GfpContentId', 'KeyPart',
                                        'ActiveViewEligibleImpression', 'TargetedCustomCriteria',
                                        'PodPosition', 'PublisherProvidedID', 'VideoPosition',
                                        'VideoFallbackPosition'])


    def preprocess(self):
        print("The shape of original NetworkBackfillImpressions log: ", self.df.shape)
        self.remove_columns()

        print("The shape of NetworkBackfillImpressions log after removing columns: ", self.df.shape)

        self.df['CustomTargeting'] = pd.Series(map(self.dictionarinize_customtargeting, self.df['CustomTargeting']))
        self.df['TimeUsec'] = pd.Series(map(self.get_utc, self.df['TimeUsec']))  # UTC
        self.df['Time'] = pd.Series(map(self.get_est, self.df['Time']))  # EST

        self.filter_product_rows()
        self.filter_customtargeting_rows()

        self.df['PageID'] = pd.Series(map(self.get_pageid_from_CT, self.df['CustomTargeting']), index=self.df.index)
        self.df['PageNo'] = pd.Series(map(self.get_pageno_from_CT, self.df['CustomTargeting']), index=self.df.index)
        self.df['AdPosition'] = pd.Series(map(self.get_pos_from_CT, self.df['CustomTargeting']), index=self.df.index)

        self.filter_non_article_rows()

        print("The shape of NetworkBackfillImpressions log after filtering some rows: ", self.df.shape)

        # print(self.df.sort_values(by=['TimeUsec']))


        unique_ids = self.df['PageID'].unique()

        # for naturalid in unique_ids:
        #     if 'blogAndPostId' not in naturalid:
        #         print(naturalid)

        print("%d unique pages in this file." % len(unique_ids))

        result_df = self.get_URIs(unique_ids)
        self.df = pd.merge(self.df, result_df, how='inner', left_on='PageID', right_on='NaturalIDs')


        self.df['URIs_pageno'] = self.df[['URIs', 'PageNo']].apply(self.process_uri, axis=1)

        self.df = self.df.drop(['PageID', 'PageNo', 'URIs'], axis=1)


        print("The shape of NetworkBackfillImpressions log after filtering by URLs: ", self.df.shape)

        # print(self.df.sort_values(by=['TimeUsec']))

        print(self.df[['URIs_pageno', 'Time', 'Country', 'Region', 'AdPosition']])


        """
        for i, row in self.df.iterrows():
            '''
            Each row is an impression
            '''
            pass
            # page_id = row['CustomTargeting']['id']
            # pos = row['CustomTargeting']['pos']
        """

    def get_URIs(self, ids):
        print('Requesting %d NatrualIDs' % len(ids))

        result_df = pd.DataFrame(columns=['naturalId', 'uri'])
        for batch in self.chunker(ids, size=180):
            api_url = ''.join([FORBES_API_ROOT + '&queryfilters=%5b%7B%22naturalId%22:%5b%22' +
                           '%22,%22'.join(batch) + '%22%5d%7D%5d&retrievedfields=id,naturalId,uri' +
                           '&limit=%d' % len(batch)])

            # print(api_url)
            response = urlopen(api_url).read().decode('utf-8')
            contentList = json.loads(response)['contentList']

            result_df = result_df.append([{key : dict[key]
                                    for key in dict if key == 'naturalId' or key == 'uri'} for dict in contentList],
                                         ignore_index=True)
            # print(result_df)
            try:
                print('Received %d/%d results' % (len(json.loads(response)['contentList']), len(batch)))
            except KeyError:
                print(json.loads(response))

        result_df.columns = ['NaturalIDs', 'URIs']
        return result_df

    def process_uri(self, x):
        # cat URI and page no
        uri = ''.join(x) + '/'
        uri = re.compile('http[s]?:\/\/www[0-9]*\.').sub('', uri)
        return uri

    def chunker(self, seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    def get_pageno_from_CT(self, customtargeting):
        if 'page' not in customtargeting:
            return ''
        return customtargeting['page']


    def get_pageid_from_CT(self, customtargeting):
        customtargeting['id'] = customtargeting['id'].replace('blogandpostid', 'blogAndPostId')
        return customtargeting['id']

    def get_pos_from_CT(self, customtargeting):
        return customtargeting['pos']

    def get_header_bids(self, customtargeting):
        return {key: customtargeting[key] for key in HEADER_BIDDING_KEYS if key in customtargeting}

    def filter_customtargeting_rows(self):
        self.df = self.df[self.df['CustomTargeting'].map(lambda row: row is not None and 'id' in row and
                                                         'pos' in row)]

    def filter_non_article_rows(self):
        self.df = self.df[self.df['PageID'].map(lambda row: 'blogAndPostId' in row)]

    def filter_product_rows(self):
        '''
        'Product': ['Ad Exchange' 'Exchange Bidding' 'First Look']
        Skip all rows whose 'Product' is 'custom_targeting'
        '''
        self.df = self.df[self.df['Product'] != 'Exchange Bidding']

    def dictionarinize_customtargeting(self, raw_customtargeting):
        if type(raw_customtargeting) is not str:
            # print(raw_customtargeting)
            return None
        return dict(x.split('=') for x in raw_customtargeting.split(';'))

    def get_utc(self, timeusec):
        return datetime.utcfromtimestamp(timeusec)  # <class 'datetime.datetime'>

    def get_est(self, time):
        return datetime.strptime(time, '%Y-%m-%d-%H:%M:%S')


if __name__ == '__main__':
    df = pd.read_csv('/Users/chong.wang/PycharmProjects/HeaderBidding/data/NetworkBackfillImpressions_330022_20180103_00', header=0, delimiter='^')
    testfile = NetworkBackfillImpressions(df)
    testfile.preprocess()