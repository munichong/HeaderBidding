import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize

root = 'I:/Desktop'
infile = os.path.join(root, 'all_predictions_factorized_shape5.csv')
outfile = os.path.join(root, 'all_predictions_factorized_shape5_predicted.csv')

class ExpectedRevenueMaximum:
    def neg_expected_revenue(self, x, row):
        not_surv_prob = 1 - np.exp(-(x / row['SCALE']) ** row['SHAPE'])
        return -((1 - not_surv_prob) * x + not_surv_prob * row['MAX_HB'])

    def predict(self, row):
        mini = minimize(self.neg_expected_revenue, np.array([0]),
                        (row), method='Nelder-Mead')

        return pd.Series({'optimal reserve price': mini.x[0],
                          'expected revenue': -mini.fun})

    def run(self, infile, outfile):
        df = pd.read_csv(infile)
        df[['optimal reserve price', 'expected revenue']] = df.apply(self.predict, axis=1)
        # for i, row in df.iterrows():
        #     res = self.predict(row)
        #     print("%f\t%d\t%f\t%f\t%f\t%f\t%f" %
        #           (row['NOT_SURV_PROB'], row['EVENTS'], row['MAX(RESERVE, REVENUE)'], row['SCALE'], row['SHAPE'],
        #            res['optimal reserve price'], res['expected revenue']))

        df.to_csv(outfile)


class SpecifyConfidence:

    CONFIDENCE = 0.3  # target survive prob

    def predict(self, row):
        return ((-1 * np.log(1 - self.CONFIDENCE)) ** (1 / row['SHAPE'])) * row['SCALE']

    def expected_revenue(self, row):
        not_surv_prob = 1 - self.CONFIDENCE
        return (1 - not_surv_prob) * row['optimal reserve price'] + not_surv_prob * row['MAX_HB']

    def run(self, infile):
        df = pd.read_csv(infile)

        df['optimal reserve price'] = df.apply(self.predict, axis=1)
        df['expected revenue'] = df.apply(self.expected_revenue, axis=1)

        for i, row in df.iterrows():
            row['optimal reserve price'] = self.predict(row)
            row['expected revenue'] = self.expected_revenue(row)
            print("%f\t%d\t%f\t%f\t%f\t%f\t%f" %
                  (row['NOT_SURV_PROB'], row['EVENTS'], row['MAX(RESERVE, REVENUE)'], row['SCALE'], row['SHAPE'],
                   row['optimal reserve price'], row['expected revenue']))

        df.to_csv(outfile)


def evaluate(predfile):
    num_fail_good_lower, num_fail_higher, num_fail_too_lower= 0, 0, 0
    num_surv_good_higher, num_surv_lower, num_surv_too_higher = 0, 0, 0
    total_fail_pred = 0
    total_surv_pred = 0
    for i, row in pd.read_csv(predfile).iterrows():
        if row['EVENTS'] == 1 and row['NOT_SURV_PROB'] >= 0.5:
            total_fail_pred += 1
            movement = (row['MAX(RESERVE, REVENUE)'] - row['optimal reserve price']) / row['MAX(RESERVE, REVENUE)']
            if 0 < movement < 0.5:
                num_fail_good_lower += 1
            elif movement < 0:
                num_fail_higher += 1
            elif movement > 0.5:
                num_fail_too_lower += 1

        elif row['EVENTS'] == 0 and row['NOT_SURV_PROB'] < 0.5:
            total_surv_pred += 1
            movement = (row['optimal reserve price'] - row['MAX(RESERVE, REVENUE)']) / row['MAX(RESERVE, REVENUE)']
            if 0 < movement < 0.5:
                num_surv_good_higher += 1
            elif movement < 0:
                num_surv_lower += 1
            elif movement > 0.5:
                num_surv_too_higher += 1

    print("num_fail_good_lower, num_fail_higher, num_fail_too_lower = %d, %d, %d" %
          (num_fail_good_lower, num_fail_higher, num_fail_too_lower))
    print("num_surv_good_higher, num_surv_lower, num_surv_too_higher = %d, %d, %d" %
          (num_surv_good_higher, num_surv_lower, num_surv_too_higher))
    print("total_fail_pred, total_surv_pred = %d, %d" % (total_fail_pred, total_surv_pred))
    print()
    print("num_fail_good_lower / total_fail_pred = %.2f%%" % (num_fail_good_lower / total_fail_pred * 100))
    print("num_fail_higher / total_fail_pred = %.2f%%" % (num_fail_higher / total_fail_pred * 100))
    print("num_fail_too_lower / total_fail_pred = %.2f%%" % (num_fail_too_lower / total_fail_pred * 100))
    print()
    print("num_surv_good_higher / total_surv_pred = %.2f%%" % (num_surv_good_higher / total_surv_pred * 100))
    print("num_surv_lower / total_surv_pred = %.2f%%" % (num_surv_lower / total_surv_pred * 100))
    print("num_surv_too_higher / total_surv_pred = %.2f%%" % (num_surv_too_higher / total_surv_pred * 100))
    print()



if __name__ == '__main__':
    ExpectedRevenueMaximum().run(infile, outfile)
    evaluate(outfile)

