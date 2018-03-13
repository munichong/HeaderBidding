import json, re, logging
import pandas as pd
import numpy as np
from datetime import datetime
from urllib.request import urlopen
from util.parameters import HEADER_BIDDING_KEYS, FORBES_API_ROOT

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class NetworkImpressions():
    def __init__(self, file_content):
        self.df = file_content

    def remove_columns(self):
        '''
       [Time, UserId, AdvertiserId, OrderId, LineItemId, CreativeId, CreativeVersion, CreativeSize,
       AdUnitId, CustomTargeting, Domain, CountryId, Country, RegionId, Region, MetroId, Metro, CityId, City,
       PostalCodeId, PostalCode, BrowserId, Browser, OSId,OS, OSVersion, BandwidthId, BandWidth,
       TimeUsec, AudienceSegmentIds, Product, RequestedAdUnitSizes, BandwidthGroupId, MobileDevice,
       MobileCapability, MobileCarrier, IsCompanion, TargetedCustomCriteria, DeviceCategory, IsInterstitial,
       EventTimeUsec2, YieldGroupNames, YieldGroupCompanyId, MobileAppId, RequestLanguage, DealId, DealType,
       AdxAccountId, SellerReservePrice, Buyer, Advertiser, Anonymous, ImpressionId]

        logger.debug(df.columns)
        '''
        self.df = self.df.drop(columns=['AdvertiserId', 'CreativeVersion', 'CreativeId',
                                        'CountryId', 'RegionId', 'MetroId', 'CityId', 'PostalCodeId',
                                        'BrowserId', 'OSId', 'BandwidthId', 'BandwidthGroupId',
                                        'EventTimeUsec2', 'DealId', 'DealType', 'AdxAccountId',
                                        'Anonymous'])

    def preprocess(self):
        logging.info("The shape of original NetworkImpressions log: (%d, %d)" % self.df.shape)
        self.remove_columns()

        logging.info("The shape of NetworkImpressions log after removing columns: (%d, %d)" % self.df.shape)

        self.df['CustomTargeting'] = pd.Series(map(self.dictionarinize_customtargeting, self.df['CustomTargeting']))
        self.df['TimeUsec'] = pd.Series(map(self.get_utc, self.df['TimeUsec']))  # UTC
        self.df['Time'] = pd.Series(map(self.get_est, self.df['Time']))  # EST

        self.filter_headerbidding_rows()
        self.filter_customtargeting_rows()

        self.df['PageID'] = pd.Series(map(self.get_pageid_from_CT, self.df['CustomTargeting']), index=self.df.index)
        self.df['PageNo'] = pd.Series(map(self.get_pageno_from_CT, self.df['CustomTargeting']), index=self.df.index)
        self.df['AdPosition'] = pd.Series(map(self.get_pos_from_CT, self.df['CustomTargeting']), index=self.df.index)

        self.filter_non_article_rows()

        logging.info("The shape of NetworkImpressions log after filtering some rows: (%d, %d)" % self.df.shape)

        unique_ids = self.df['PageID'].unique()

        logging.info("%d unique pages in this file." % len(unique_ids))

        result_df = self.get_URIs(unique_ids)
        self.df = pd.merge(self.df, result_df, how='inner', left_on='PageID', right_on='NaturalIDs')

        self.df['URIs_pageno'] = self.df[['URIs', 'PageNo']].apply(self.process_uri, axis=1)

        self.df = self.df.drop(['PageNo', 'URIs'], axis=1)

        logging.info("The shape of NetworkImpressions log after filtering by URLs: (%d, %d)" % self.df.shape)



    def get_URIs(self, ids):
        logging.info('Requesting %d NatrualIDs' % len(ids))

        result_df = pd.DataFrame(columns=['naturalId', 'uri'])
        for batch in self.chunker(ids, size=180):
            api_url = ''.join([FORBES_API_ROOT + '&queryfilters=%5b%7B%22naturalId%22:%5b%22' +
                           '%22,%22'.join(batch) + '%22%5d%7D%5d&retrievedfields=id,naturalId,uri' +
                           '&limit=%d' % len(batch)])

            response = urlopen(api_url).read().decode('utf-8')
            contentList = json.loads(response)['contentList']

            result_df = result_df.append([{key : dict[key]
                                    for key in dict if key == 'naturalId' or key == 'uri'} for dict in contentList],
                                         ignore_index=True)
            try:
                logging.info('Received %d/%d results' % (len(json.loads(response)['contentList']), len(batch)))
            except KeyError:
                logging.warning(json.loads(response))
            # break

        result_df.columns = ['NaturalIDs', 'URIs']
        return result_df

    def chunker(self, seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    def process_uri(self, x):
        # cat URI and page no
        if x[1]:
            uri = ''.join(x) + '/'
        else:
            uri = x[0]
        uri = re.compile('http[s]?:\/\/www[0-9]*\.').sub('', uri)
        return uri

    def filter_non_article_rows(self):
        self.df = self.df[self.df['PageID'].map(lambda row: 'blogAndPostId' in row)]

    def get_pageno_from_CT(self, customtargeting):
        if 'page' not in customtargeting:
            return ''
        return customtargeting['page']


    def get_pageid_from_CT(self, customtargeting):
        customtargeting['id'] = customtargeting['id'].replace('blogandpostid', 'blogAndPostId')
        return customtargeting['id']

    def get_pos_from_CT(self, customtargeting):
        return customtargeting['pos']

    def filter_headerbidding_rows(self):
        pass

    def filter_customtargeting_rows(self):
        self.df = self.df[self.df['CustomTargeting'].map(lambda row: row is not None and 'id' in row and
                                                         'pos' in row)]

    def dictionarinize_customtargeting(self, raw_customtargeting):
        if type(raw_customtargeting) is not str:
            return None
        return dict(x.split('=') for x in raw_customtargeting.split(';'))

    def get_utc(self, timeusec):
        return datetime.utcfromtimestamp(timeusec)  # <class 'datetime.datetime'>

    def get_est(self, time):
        return datetime.strptime(time, '%Y-%m-%d-%H:%M:%S')



if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    df = pd.read_csv('M:/Research Datasets/Header Bidding Data/NetworkBackfillImpressions/2018.01.03/NetworkImpressions_330022_20180103_00', header=0, delimiter=',')
    testfile = NetworkImpressions(df)
    testfile.preprocess()