import pandas as pd
import numpy as np

# Load data
target = pd.read_csv('Data/predictive_model_target_sharpes.csv', index_col=0)
# Remove the first 3 rows (contain NaNs)
target = target.drop(index=[1,2,3]).reset_index(drop=True)

# Calculate the target variable
target['Max Sharpe'] = target[['sharpe_ratio_1n', 'sharpe_ratio_rp', 'sharpe_ratio_mv']].idxmax(axis=1)
target['Max Return'] = target[['monthly_r_1n', 'monthly_r_rp', 'monthly_r_mv']].idxmax(axis=1)
target['Min Volatility'] = target[['monthly_vol_1n', 'monthly_vol_rp', 'monthly_vol_mv']].idxmin(axis=1)

# Map the text entries to numbers
# 0 : 1/n, 1: risk parity, 2: mean variance
target['Max Sharpe'] = target['Max Sharpe'].replace({'sharpe_ratio_1n':0, 'sharpe_ratio_rp':1, 'sharpe_ratio_mv':2})
target['Max Return'] = target['Max Return'].replace({'monthly_r_1n':0, 'monthly_r_rp':1, 'monthly_r_mv':2})
target['Min Volatility'] = target['Min Volatility'].replace({'monthly_vol_1n':0, 'monthly_vol_rp':1, 'monthly_vol_mv':2})

# For each target variable, print the number of samples for each portfolio type
print("Target Variable: Max Sharpe")
print("1/n: ", target[target['Max Sharpe']==0].shape[0])
print("rp: ",target[target['Max Sharpe']==1].shape[0])
print("mv: ",target[target['Max Sharpe']==2].shape[0])

print("Target Variable: Max Return")
print("1/n: ", target[target['Max Return']==0].shape[0])
print("rp: ",target[target['Max Return']==1].shape[0])
print("mv: ",target[target['Max Return']==2].shape[0])

print("Target Variable: Min Volatility")
print("1/n: ", target[target['Min Volatility']==0].shape[0])
print("rp: ",target[target['Min Volatility']==1].shape[0])
print("mv: ",target[target['Min Volatility']==2].shape[0])

# Select the 'Max Sharpe' target variable
target = target[['month_start_date', 'month_end_date', 'Max Sharpe']]
# Save the target variables dataframe to a csv
target.to_csv('Data/target_predictive.csv')