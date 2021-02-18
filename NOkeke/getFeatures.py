import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def applyPCA(returns_df):
    '''
    Applies PCA to a given dataframe
    :param returns_df: dataframe of 3 months of asset returns
    :return: percentage of variance explained for each of the eigenvectors of the dataframe
    '''
    pca = PCA()
    returns_pca = pca.fit(returns_df)
    percent_var = returns_pca.explained_variance_ratio_
    return percent_var

def getEigenReq(returns_df, list_of_percents):
    '''
    Applies PCA to the given dataframe and determines the number of eigenvectors required to explain
    each percentage of variability in 'list_of_percents'
    :param returns_df: dataframe of 3 months of asset returns
    :return: all_num_eigen: list of number of eigenvectors required to explain each percentage of variability
    '''
    all_num_eigen = []
    # Iterate through each percentage and determine the number of eigenvectors needed to explain that
    # percentage of variability
    for percent in list_of_percents:
        percent_var = applyPCA(returns_df)
        num_eigen = 0
        total_variability = 0
        # While total variability is less than the target percentage
        while(total_variability < percent):
            # Increment the number of eigenvectors and add to the total variability
            total_variability = total_variability + percent_var[num_eigen]
            num_eigen = num_eigen + 1
        all_num_eigen = all_num_eigen + [num_eigen]
    return all_num_eigen

if __name__ == "__main__":

    # Load asset data
    asset_prices = pd.read_csv('Data/US_set.csv', index_col=0)

    # Compute daily log returns
    asset_returns = np.log(asset_prices.loc[:,'LargeCap':'SUGAR']/asset_prices.loc[:,'LargeCap':'SUGAR'].shift(1))
    # Add the date column back to the dataframe so it can be used to
    # determine indices that indicate the start of each month
    asset_returns = pd.concat([asset_prices['Date'], asset_returns], axis=1)
    # Drop the first row because all entries are 'NaN'
    asset_returns = asset_returns.drop(index=0)

    # Get a list of the indices that correspond to the start of each month
    # Initialize the list of indices to start with 1 (index of the row with the returns for the first day in Jan 2001)
    month_start_indices = [1]
    # Initialize month to Jan
    month = '01'
    # Iterate through each row in the asset_returns dataframe
    for i in asset_returns.index.to_list():
        # If the month in row i is not the value stored in 'month', then the month has changed
        if month != asset_returns.loc[i, 'Date'][5:7]:
            # Add the row number to the month_start indices
            month_start_indices = month_start_indices + [i]
            # Update 'month' to the new month
            month = asset_returns.loc[i, 'Date'][5:7]

    # Remove the last entry in 'month_start_indices' because the last target variable is for June 2019,
    # therefore since it's a predictive model features are created using data up to and including May 2019.
    # So the last entry (index for July 2019) is omitted
    month_start_indices = month_start_indices[:len(month_start_indices)-1]

    '''
    Calculate features
    '''
    # Initialize a dataframe to hold the values for each type of feature
    # feature-month_end_date: last day in the 3-month period used to calculate the features
    # target-month_start_date: first day in the 1-month period for the target associated with the features
    feat1 = pd.DataFrame(columns=['feature-month_end_date','target-month_start_date']+['feat1_'+str(i) for i in range(0,10)])
    feat2 = pd.DataFrame(
        columns=['feature-month_end_date', 'target-month_start_date'] + ['feat2_0.' + str(i) for i in range(3, 10)])
    feat3 = pd.DataFrame(
        columns=['feature-month_end_date', 'target-month_start_date','feat3_eq_0', 'feat3_eq_1', 'feat3_bd_0',
                 'feat3_bd_1', 'feat3_com_0', 'feat3_com_1'])

    # Iterate through all the indices that mark the start of each month
    for i in range(0, len(month_start_indices)-3, 1):
        # Create a separate dataframe with 3 months of daily returns
        three_month_rets = asset_returns.loc[month_start_indices[i]:month_start_indices[i+3]-1, :]

        '''
        Feature #1: What % of the variance is explained by the top 10 eigenvectors?
        '''
        percent_var = applyPCA(three_month_rets.loc[:,'LargeCap':'SUGAR'])
        top10 = percent_var[0:10]

        # Store the features in feat1
        feat1.loc[i,'feature-month_end_date'] = asset_returns.loc[month_start_indices[i+3]-1, 'Date']
        feat1.loc[i,'target-month_start_date'] = asset_returns.loc[month_start_indices[i+3], 'Date']
        feat1.loc[i, 'feat1_0':] = top10

        '''
        Feature #2: How many eigenvectors are required to explain 30%, 40%, 50%, 60%, 70%, 80%, 90% of variability
        '''
        num_eigenvectors = getEigenReq(three_month_rets.loc[:,'LargeCap':'SUGAR'], [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        # Store the features in feat2
        feat2.loc[i, 'feature-month_end_date'] = asset_returns.loc[month_start_indices[i + 3] - 1, 'Date']
        feat2.loc[i, 'target-month_start_date'] = asset_returns.loc[month_start_indices[i + 3], 'Date']
        feat2.loc[i, 'feat2_0.3':] = num_eigenvectors

        '''
        Feature #3: What % of variance is explained by the top 2 eigenvectors for each subset?
        Subsets: Equities, Bonds, Commodities
        '''
        Equities = ['LargeCap','SmallCap','Financials','Energy','Materials','Industrials','Info Tech',
                    'Cons. Discretionary','Health Care','Telecom','Cons. Staples','Utilities']
        Bonds = ['US Aggregate','US Treasury','US Corporate','US High Yield']
        Commodities = ['BCOM Index','BRENT CRUDE','WTI CRUDE','GOLD ','SILVER','CORN','SUGAR']

        # Create dataframes for each subset
        three_month_rets_eq = three_month_rets[Equities]
        three_month_rets_bd = three_month_rets[Bonds]
        three_month_rets_com = three_month_rets[Commodities]

        # Compute features
        percent_var_eq = applyPCA(three_month_rets_eq)
        top2_eq = percent_var_eq[0:2]
        percent_var_bd = applyPCA(three_month_rets_bd)
        top2_bd = percent_var_bd[0:2]
        percent_var_com = applyPCA(three_month_rets_com)
        top2_com = percent_var_com[0:2]

        # Store the features in feat3
        feat3.loc[i, 'feature-month_end_date'] = asset_returns.loc[month_start_indices[i + 3] - 1, 'Date']
        feat3.loc[i, 'target-month_start_date'] = asset_returns.loc[month_start_indices[i + 3], 'Date']
        feat3.loc[i, 'feat3_eq_0':'feat3_eq_1'] = top2_eq
        feat3.loc[i, 'feat3_bd_0':'feat3_bd_1'] = top2_bd
        feat3.loc[i, 'feat3_com_0':'feat3_com_1'] = top2_com

    '''
    Save dataframe containing all features
    '''
    features = feat1.merge(feat2, how='outer', on=['feature-month_end_date', 'target-month_start_date'])
    features = features.merge(feat3, how='outer', on=['feature-month_end_date', 'target-month_start_date'])

    features.to_csv('Data/features.csv')