import pandas as pd
import numpy as np
from sklearn.decomposition import train_test_split

# Load targets and features
# Merge them together on 'target-month_start_date'
# features[target[Max Sharpe] == 0]
# Check that dates match after?
# features_ones[target_ones['Date] != features_ones['Date']]
# Split them into 3 dataframes, 1 for each class
# Run train test split to split each subdataframe into training and testing sets
# Concatenate the 3 training sets together (and the 3 testing sets)
# Resample the training and testing sets to mix them up
# Split the training and testing sets into features and targets
# Save the training and testing sets