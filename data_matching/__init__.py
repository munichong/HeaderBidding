import pandas as pd

orderId_df = pd.ExcelFile('../header bidder.xlsx').parse(0)
print(orderId_df)
print(orderId_df.shape)

print(orderId_df[orderId_df.iloc[:,3].map(lambda row: row == 'bidder')].iloc[:,2].shape)
headerbiddingIds = set(orderId_df[orderId_df.iloc[:,3].map(lambda row: row == 'bidder')].iloc[:,2].values)
print(len(headerbiddingIds))