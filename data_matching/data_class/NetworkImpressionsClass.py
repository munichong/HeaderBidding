import logging
import pandas as pd

from data_matching.data_class.DFPDataClass import DFPData


logger = logging.getLogger()
logger.setLevel(logging.INFO)

hb_orderIds_path = '../header bidder.xlsx'

class NetworkImpressions(DFPData):
    def __init__(self, file_content):
        super().__init__(file_content)

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

        self.filter_headerbidding_rows()

        self.df['CustomTargeting'] = pd.Series(map(self.dictionarinize_customtargeting, self.df['CustomTargeting']),
                                               index=self.df.index)
        self.filter_customtargeting_rows()

        self.df['TimeUsec'] = pd.Series(map(self.get_utc, self.df['TimeUsec']), index=self.df.index)  # UTC
        self.df['Time'] = pd.Series(map(self.get_est, self.df['Time']), index=self.df.index)  # EST


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


    def filter_headerbidding_rows(self):
        orderId_df = pd.ExcelFile(hb_orderIds_path).parse(0)
        headerbiddingIds = set(orderId_df[orderId_df.iloc[:, 3].map(lambda row: row == 'bidder')].iloc[:, 2].values)
        self.df = self.df[self.df['OrderId'].map(lambda row: row in headerbiddingIds)]



if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    df = pd.read_csv('M:/Research Datasets/Header Bidding Data/NetworkBackfillImpressions/2018.01.03/NetworkImpressions_330022_20180103_00', header=0, delimiter=',')
    testfile = NetworkImpressions(df)
    testfile.preprocess()