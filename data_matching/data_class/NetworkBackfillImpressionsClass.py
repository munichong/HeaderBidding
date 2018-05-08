import logging
import pandas as pd

from util.parameters import HEADER_BIDDING_KEYS
from data_matching.data_class.DFPDataClass import DFPData


logger = logging.getLogger()
logger.setLevel(logging.INFO)

# HEADER_BIDDING_KEYS = ('mnetbidprice',
#                        'mnet_abd',
#                        'mnet_fbcpm',
#                        'amznbid',
#                        'fb_bid_price_cents')
#
# FORBES_API_ROOT = 'https://forbesapis.forbes.com/forbesapi/content/all.json/?code=a6016ad7796e2165bba73787d68f3162b29f9bd7'

class NetworkBackfillImpressions(DFPData):
    def __init__(self, file_content):
        super().__init__(file_content)


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
       'TimeUsec2', 'VideoPosition', 'VideoFallbackPosition'，

       'RefererURL'， 'AudienceSegmentIds'， 'MobileDevice'， 'OSVersion'， 'MobileCapability'，
       'MobileCarrier'， 'IsCompanion'， 'BandwidthGroupId'， 'EventTimeUsec2'， 'IsInterstitial'，
       'EventKeyPart'， 'EstimatedBackfillRevenue'， 'YieldGroupNames'， 'YieldGroupCompanyId'，
       'MobileAppId'， 'RequestLanguage'， 'DealId', 'DealType', 'AdxAccountId', 'SellerReservePrice',
       'Buyer', 'Advertiser', 'Anonymous', 'ImpressionId']

        logger.debug(df.columns)
        '''
        self.df = self.df.drop(columns=['IP', 'CreativeVersion', 'Domain',
                                        'CountryId', 'RegionId', 'MetroId', 'CityId', 'PostalCodeId',
                                        'BrowserId', 'OSId', 'BandwidthId', 'GfpContentId', 'KeyPart',
                                        'ActiveViewEligibleImpression', 'TargetedCustomCriteria',
                                        'PodPosition', 'PublisherProvidedID', 'VideoPosition',
                                        'VideoFallbackPosition', 'YieldGroupNames', 'YieldGroupCompanyId',
                                        'DealId', 'DealType', 'Anonymous'])


    def preprocess(self):
        logging.info("The shape of original NetworkBackfillImpressions log: (%d, %d)" % self.df.shape)
        self.remove_columns()

        logging.info("The shape of NetworkBackfillImpressions log after removing columns: (%d, %d)" % self.df.shape)

        self.filter_product_rows()
        # print(self.df)

        self.df['CustomTargeting'] = pd.Series(map(self.dictionarinize_customtargeting, self.df['CustomTargeting']),
                                               index=self.df.index)

        # print(self.df)
        self.filter_customtargeting_rows()


        self.df['TimeUsec'] = pd.Series(map(self.get_utc, self.df['TimeUsec']), index=self.df.index)  # UTC
        self.df['Time'] = pd.Series(map(self.get_est, self.df['Time']), index=self.df.index)  # EST


        self.df['PageID'] = pd.Series(map(self.get_pageid_from_CT, self.df['CustomTargeting']), index=self.df.index)
        self.df['PageNo'] = pd.Series(map(self.get_pageno_from_CT, self.df['CustomTargeting']), index=self.df.index)
        self.df['AdPosition'] = pd.Series(map(self.get_pos_from_CT, self.df['CustomTargeting']), index=self.df.index)

        self.filter_non_article_rows()

        logging.info("The shape of NetworkBackfillImpressions log after filtering some rows: (%d, %d)" % self.df.shape)

        # logger.debug(self.df.sort_values(by=['TimeUsec']))


        unique_ids = self.df['PageID'].unique()

        # for naturalid in unique_ids:
        #     if 'blogAndPostId' not in naturalid:
        #         logger.debug(naturalid)

        logging.info("%d unique pages in this file." % len(unique_ids))

        result_df = self.get_URIs(unique_ids)
        self.df = pd.merge(self.df, result_df, how='inner', left_on='PageID', right_on='NaturalIDs')

        self.df['URIs_pageno'] = self.df[['URIs', 'PageNo']].apply(self.process_uri, axis=1)

        self.df = self.df.drop(['PageNo', 'URIs'], axis=1)

        logging.info("The shape of NetworkBackfillImpressions log after filtering by URLs: (%d, %d)" % self.df.shape)


    def get_header_bids(self, customtargeting):
        return {key: customtargeting[key] for key in HEADER_BIDDING_KEYS if key in customtargeting}

    def filter_product_rows(self):
        '''
        'Product': ['Ad Exchange' 'Exchange Bidding' 'First Look']
        Skip all rows whose 'Product' is 'custom_targeting'
        '''
        self.df = self.df[self.df['Product'] != 'Exchange Bidding']

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    df = pd.read_csv('M:/Research Datasets/Header Bidding Data/NetworkBackfillImpressions/2018.01.03/NetworkBackfillImpressions_330022_20180103_00', header=0, delimiter='^')
    testfile = NetworkBackfillImpressions(df)
    testfile.preprocess()