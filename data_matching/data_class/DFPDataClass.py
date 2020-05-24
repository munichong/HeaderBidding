import json
import logging
import re
import time
from datetime import datetime
from urllib.request import urlopen

import pandas as pd

from util.parameters import FORBES_API_ROOT

logger = logging.getLogger()
logger.setLevel(logging.INFO)

hb_orderIds_path = '../header bidder.xlsx'


class DFPData():
    def __init__(self, file_content):
        self.df = file_content

    def get_URIs(self, ids):
        logging.info('Requesting %d NatrualIDs' % len(ids))

        result_df = pd.DataFrame(columns=['naturalId', 'uri'])
        for batch in self.chunker(ids, size=180):
            api_url = ''.join([FORBES_API_ROOT + '&queryfilters=%5b%7B%22naturalId%22:%5b%22' +
                               '%22,%22'.join(batch) + '%22%5d%7D%5d&retrievedfields=id,naturalId,uri' +
                               '&limit=%d' % len(batch)])

            # logger.debug(api_url)
            response = urlopen(api_url).read().decode('utf-8')
            contentList = json.loads(response)['contentList']

            result_df = result_df.append([{key: dict[key]
                                           for key in dict if key == 'naturalId' or key == 'uri'} for dict in
                                          contentList],
                                         ignore_index=True)
            # logger.debug(result_df)
            try:
                logging.info('Received %d/%d results' % (len(json.loads(response)['contentList']), len(batch)))
            except KeyError:
                logging.warning(json.loads(response))

            time.sleep(0.5)

        result_df.columns = ['NaturalIDs', 'URIs']
        return result_df

    def process_uri(self, x):
        # cat URI and page no
        if x[1]:
            uri = ''.join(x) + '/'
        else:
            uri = x[0]
        uri = re.compile('http[s]?:\/\/www[0-9]*\.').sub('', uri)
        return uri

    def chunker(self, seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    def get_pageno_from_CT(self, customtargeting):
        if 'page' not in customtargeting:
            return ''
        return customtargeting['page']

    def get_pageid_from_CT(self, customtargeting):
        try:
            customtargeting['id'] = customtargeting['id'].replace('blogandpostid', 'blogAndPostId')
        except AttributeError:
            print(customtargeting['id'])
        return customtargeting['id']

    def get_pos_from_CT(self, customtargeting):
        return customtargeting['pos']

    def filter_non_article_rows(self):
        self.df = self.df[self.df['PageID'].map(lambda row: 'blogAndPostId' in row)]

    def filter_customtargeting_rows(self):
        self.df = self.df[self.df['CustomTargeting'].map(lambda row: row is not None and 'id' in row and
                                                                     'pos' in row)]

    def dictionarinize_customtargeting(self, raw_customtargeting):
        if type(raw_customtargeting) is not str:
            return None
        targeting_dict = {}
        for x in raw_customtargeting.split(';'):
            key, value = x.split('=')
            # value = self.parse_str_value(key, value)
            if key in targeting_dict:
                if type(targeting_dict[key]) is list:
                    targeting_dict[key].append(value)
                else:
                    targeting_dict[key] = [targeting_dict[key], value]
            else:
                targeting_dict[key] = value
        return targeting_dict

    def parse_str_value(self, key, value):
        if key in ['id', 'channel', 'section', 'displaychannel', 'displaysection', 'pos']:
            return value
        if value.lower() == "true" or value.lower() == "false":
            return bool(value)
        try:
            return float(value)
        except ValueError:
            return value

    def get_utc(self, timeusec):
        return datetime.utcfromtimestamp(timeusec)  # <class 'datetime.datetime'>

    def get_est(self, time):
        return datetime.strptime(time, '%Y-%m-%d-%H:%M:%S')
