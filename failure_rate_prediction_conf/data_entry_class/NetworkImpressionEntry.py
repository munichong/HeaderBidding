from failure_rate_prediction_conf.data_entry_class.ImpressionEntry import ImpressionEntry

class NetworkImpressionEntry(ImpressionEntry):
    def __init__(self, doc):
        super().__init__(doc)

    def get_target(self):
        target = []

        ''' Duration '''
        if not self.is_qualified():
            return None
        duration = self.get_floor_price()
        target.append(duration)

        ''' Event '''
        target.append(1)

        return target

    def get_floor_price(self):
        highest_header_bid = self.get_highest_header_bid()
        if not highest_header_bid:
            return None
        return self.to_closest_5cents(highest_header_bid)

    def get_highest_header_bid(self):
        header_bids = self.get_headerbids()
        if not header_bids:
            return None
        return max(v for v in header_bids if v is not None)

    def to_closest_5cents(self, num):
        return num - (num % 0.05)

    def is_qualified(self):
        if not self.has_headerbidding():  # header-bidding-won impression must contain header bids.
            return False
        return True
