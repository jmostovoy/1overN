import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load targets and features
features = pd.read_csv('Data/features.csv', index_col=0)
target = pd.read_csv('Data/target_predictive.csv', index_col=0)

# Split target into 3 dataframes, 1 for each class
target_0s = target[target['Max Sharpe'] == 0]
target_1s = target[target['Max Sharpe'] == 1]
target_2s = target[target['Max Sharpe'] == 2]

# Split features into 3 dataframes, matching the target dataframes
features_0s = features[features['target-month_start_date'].eq(target_0s['month_start_date'])]
features_1s = features[features['target-month_start_date'].eq(target_1s['month_start_date'])]
features_2s = features[features['target-month_start_date'].eq(target_2s['month_start_date'])]

# Split each sub-dataframe into training and testing sets
feat_0s_train, feat_0s_test, targ_0s_train, targ_0s_test = train_test_split(features_0s, target_0s, test_size=0.3, random_state=0)
feat_1s_train, feat_1s_test, targ_1s_train, targ_1s_test = train_test_split(features_1s, target_1s, test_size=0.3, random_state=0)
feat_2s_train, feat_2s_test, targ_2s_train, targ_2s_test = train_test_split(features_2s, target_2s, test_size=0.3, random_state=0)

# Concatenate the 3 training sets together
features_train = pd.concat([feat_0s_train, feat_1s_train, feat_2s_train], axis=0)
target_train = pd.concat([targ_0s_train, targ_1s_train, targ_2s_train], axis=0)

# Concatenate the training features and targets together so they can be shuffled
train = pd.concat([features_train, target_train], axis=1)

# Concatenate the 3 testing sets together
features_test = pd.concat([feat_0s_test, feat_1s_test, feat_2s_test], axis=0)
target_test = pd.concat([targ_0s_test, targ_1s_test, targ_2s_test], axis=0)

# Concatenate the testing features and targets together so they can be shuffled
test = pd.concat([features_test, target_test], axis=1)

# Resample the training and testing sets to mix them up
train = train.sample(frac=1, random_state=0)
test = test.sample(frac=1, random_state=0)

# Split the training and testing sets into features and targets
features_train = train.loc[:,'feat1_0':'feat6_com_min']
target_train = train.loc[:, 'Max Sharpe']

features_test = test.loc[:,'feat1_0':'feat6_com_min']
target_test = test.loc[:, 'Max Sharpe']

# Save the training and testing sets
features_train.to_csv('Data/features_train.csv')
features_test.to_csv('Data/features_test.csv')
target_train.to_csv('Data/target_train.csv')
target_test.to_csv('Data/target_test.csv')