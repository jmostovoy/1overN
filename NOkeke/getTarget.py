import pandas as pd
import numpy as np

# Load data
target = pd.read_csv('Data/predictive_model_target_sharpes.csv', index_col=0)
# Remove the first 3 rows (contain NaNs)
target = target.drop(index=[1,2,3]).reset_index()

# Calculate the target variable
target['Max Sharpe'] = target[['sharpe_ratio_1n', 'sharpe_ratio_rp', 'sharpe_ratio_mv']].idxmax(axis=1)
target['Max Return'] = target[['monthly_r_1n', 'monthly_r_rp', 'monthly_r_mv']].idxmax(axis=1)
target['Min Volatility'] = target[['monthly_vol_1n', 'monthly_vol_rp', 'monthly_vol_mv']].idxmin(axis=1)

target['Max Sharpe'] = target['Max Sharpe'].replace({'sharpe_ratio_1n':0, 'sharpe_ratio_rp':1, 'sharpe_ratio_mv':2})
target['Max Return'] = target['Max Return'].replace({'monthly_r_1n':0, 'monthly_r_rp':1, 'monthly_r_mv':2})
target['Min Volatility'] = target['Min Volatility'].replace({'monthly_vol_1n':0, 'monthly_vol_rp':1, 'monthly_vol_mv':2})

print(target[target['Min Volatility']==0].shape[0])
print(target[target['Min Volatility']==1].shape[0])
print(target[target['Min Volatility']==2].shape[0])

print(target.head())