import pandas as pd, re, pycountry
from datetime import datetime
from urllib.parse import urlparse
from pprint import pprint


class AdExchangeImpressions():
    def __init__(self, file_content):
        self.df = file_content
        self.df.columns = ['transaction_id', 'date_time', 'is_anonymous', 'winning_buyer', 'site_name', 'anonymous_id',
                           'request_url', 'allow_data_collection_for_interest_targeting', 'allow_user_list_targeting',
                           'allow_user_interest_targeting', 'matched_geo_list', 'matched_verticals_list', 'channels',
                           'ad_type', 'client_ip', 'adslot_code', 'publisher_revenue_usd', 'seller_reserve_price_usd',
                           'was_userlist_targeted', 'was_match_on_keyword', 'num_clicks', 'transaction_type',
                           'deal_ids', 'buyer_network_names', 'advertiser']

    def remove_columns(self):
        self.df = self.df.drop(columns=['transaction_id', 'is_anonymous', 'anonymous_id',
                                        'matched_verticals_list', 'client_ip', 'deal_ids',
                                        "allow_user_list_targeting", "allow_user_interest_targeting",
                                        "allow_data_collection_for_interest_targeting"])

    def filter_missing_val_rows(self, col_name):
        self.df = self.df[self.df[col_name].notnull()]

    def extract_ad_positions(self, channels):
        if not channels or type(channels) is float:
            return []
        pieces = re.split("[ |\+]", channels)
        return [s[s.index('pos=') + 4:] for s in pieces if 'pos=' in s]

    def channel2adpositions(self, raw_channel):
        ad_positions = self.extract_ad_positions(raw_channel)
        # if has multiple ad positions
        if len(set(ad_positions)) != 1:
            return None
        return ad_positions[0]

    def expand_geo(self, geo_loc):
        if not geo_loc or type(geo_loc) is not str:
            return '', ''

        country_name = ''
        if '-' in geo_loc:
            # first, try to get geo full name by the entire geo_loc string
            subdivision_instance = pycountry.subdivisions.get(code=geo_loc)
            if subdivision_instance:
                country_name, state_name = pycountry.countries.get(alpha_2=subdivision_instance.country_code).name, subdivision_instance.name
                return country_name, state_name
            else:
                geo_loc_split = geo_loc.split('-')
                try:
                    country_name = pycountry.countries.get(alpha_2=geo_loc_split[0]).name
                except KeyError:
                    print("Country: %s is not found." % geo_loc)
                    return '', ''
                print("State: %s is not found." % geo_loc)
                return country_name, ''
        else:
            try:
                country_name = pycountry.countries.get(alpha_2=geo_loc).name
            except KeyError:
                print("Country: %s is not found." % geo_loc)
                return '', ''
            return country_name, ''


    def preprocess(self):
        print("The shape of original Ad Exchange log: ", self.df.shape)
        self.remove_columns()

        print("The shape of Ad Exchange log after removing columns:", self.df.shape)

        self.df['cleaned_url'] = pd.Series(map(self.clean_url, self.df['request_url']))
        self.df['date_time'] = pd.Series(map(self.get_pacific_time, self.df['date_time']))
        self.df['channels'] = pd.Series(map(self.channel2adpositions, self.df['channels']))
        self.df.loc[:, 'publisher_revenue_usd'] *= 1000

        self.filter_missing_val_rows('channels')
        self.filter_missing_val_rows('matched_geo_list')


        country_list, state_list = zip(*map(self.expand_geo, self.df['matched_geo_list']))
        print(len(country_list), len(self.df))
        assert len(country_list) == len(self.df) and len(state_list) == len(self.df)

        self.df['expanded_country'], self.df['expanded_state'] = pd.Series(country_list, index=self.df.index), pd.Series(state_list, index=self.df.index)

        self.filter_missing_val_rows('expanded_country')

        print("The shape of Ad Exchange log after filtering out missing value rows:", self.df.shape)

        print(self.df[['cleaned_url', 'date_time', 'expanded_country', 'expanded_state', 'channels']])
        # print(self.df.dtypes)


    def clean_url(self, raw_url, ):
        url = self.remove_url_parameters(raw_url)
        # https://www3.forbes.com/sites/davidparnell/2017
        # regularize 'https'-->'http' and 'www3'-->'www'
        # results:
        # forbes.com/business/homes-in-americas-25-most-expensive-zip-codes-2016/25/
        # http://quiz.forbes.com/the-ultimate-game-of-throne-quiz/2/
        url = re.compile('http[s]?:\/\/www[0-9]*\.').sub('', url)
        # print(url)
        return url



    def remove_url_parameters(self, raw_url):
        ''' remove parameters in the raw_url
            but keep page numbers '''
        parse_result = urlparse(raw_url)
        clean_url = '{0}://{1}{2}'.format(parse_result.scheme, parse_result.netloc, parse_result.path)
        #     print("clean_url:", clean_url)
        return clean_url

    def get_pacific_time(self, date_time):
        return datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')  # <class 'datetime.datetime'>



if __name__ == '__main__':
    df = pd.read_csv('/Users/chong.wang/PycharmProjects/HeaderBidding/data/REPORT_CSV_seller_IC_87371021_2018-01-04-00000-of-00010', delimiter=',')
    testfile = AdExchangeImpressions(df)
    testfile.preprocess()