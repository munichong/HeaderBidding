import pandas as pd
import numpy as np
from scipy.optimize import minimize


infile = 'output/all_predictions_factorized.csv'
outfile = 'output/all_predictions_factorized_predicted.csv'

class ExpectedRevenueMaximum:
    def neg_expected_revenue(self, x, scale):
        return -x * (np.exp(-(x / scale) ** 0.2))

    def predict(self, scale):
        mini = minimize(self.neg_expected_revenue, np.array([0]), (scale), method='Nelder-Mead')

        return pd.Series({'optimal reserve price': mini.x[0],
                          'expected revenue': -mini.fun})

    def run(self, infile, outfile):
        df = pd.read_csv(infile)
        # df[['optimal reserve price', 'expected revenue']] = df['SCALE'].apply(self.predict)
        for i, row in df.iterrows():
            res = self.predict(row.loc['SCALE'])
            print(row['NOT_SURV_PROB'], row['EVENTS'], row['MAX(RESERVE, REVENUE)'], row['SCALE'], res['optimal reserve price'], res['expected revenue'])

        df.to_csv(outfile)


class SpecifyConfidence:
    def predict(self, scale):
        return (np.log(0.3) * -1) ** 5 * scale

    def run(self, infile):
        df = pd.read_csv(infile)

        for i, row in df.iterrows():
            res = self.predict(row.loc['SCALE'])
            print(row['NOT_SURV_PROB'], row['EVENTS'], row['MAX(RESERVE, REVENUE)'], row['SCALE'], res)


if __name__ == '__main__':
    ExpectedRevenueMaximum().run(infile, outfile)
