import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Load data
features_train = pd.read_csv('Data/features_train.csv', index_col=0)
features_test = pd.read_csv('Data/features_test.csv', index_col=0)
target_train = pd.read_csv('Data/target_train.csv', index_col=0)
target_test = pd.read_csv('Data/target_test.csv', index_col=0)

# Binarize the labels
target_train = label_binarize(target_train.values, classes=[0, 1, 2])
target_test = label_binarize(target_test.values, classes=[0, 1, 2])

# Make a list converting class number to class name, for plotting
class_to_num = {0: '1/N', 1: 'RP', 2: 'MVO'}

# '''
# Logistic Regression Classifier
# '''
# # Fit classifier
# logreg = OneVsRestClassifier(LogisticRegression(max_iter=250))
# logreg.fit(features_train.values, target_train)
# logreg_score = logreg.decision_function(features_test)
#
# # Initialize dictionaries
# precision = {}
# recall = {}
#
# # Plot the PR curve for each class
# for i in range(3):
#     precision[i], recall[i], _ = precision_recall_curve(target_test[:, i],
#                                                         logreg_score[:, i])
#     plt.figure()
#     plt.step(recall[i], precision[i], where='post')
#
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.ylim([0.0, 1.05])
#     plt.xlim([0.0, 1.0])
#     plt.title('Logistic Regression Precision-Recall Curve; Class: '+class_to_num[i])
#     plt.show()
#
# '''
# MLP Classifier
# '''
# # Fit classifier
# mlp = OneVsRestClassifier(MLPClassifier(max_iter=2000, random_state=0))
# mlp.fit(features_train.values, target_train)
# mlp_score = mlp.predict_proba(features_test)
#
# # Initialize dictionaries
# precision = {}
# recall = {}
#
# # Plot the PR curve for each class
# for i in range(3):
#     precision[i], recall[i], _ = precision_recall_curve(target_test[:, i],
#                                                         mlp_score[:, i])
#     plt.figure()
#     plt.step(recall[i], precision[i], where='post')
#
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.ylim([0.0, 1.05])
#     plt.xlim([0.0, 1.0])
#     plt.title('MLP Precision-Recall Curve; Class: '+class_to_num[i])
#     plt.show()

'''
Random Forest Classifier
'''
# Fit classifier
rf = OneVsRestClassifier(RandomForestClassifier())
rf.fit(features_train.values, target_train)
rf_score = rf.predict_proba(features_test)

# Initialize dictionaries
precision = {}
recall = {}

# Plot the PR curve and feature importances for each class
for i in range(3):
    precision[i], recall[i], _ = precision_recall_curve(target_test[:, i],
                                                        rf_score[:, i])
    plt.figure()
    plt.step(recall[i], precision[i], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Random Forest Precision-Recall Curve; Class: '+class_to_num[i])
    plt.show()

    feat_importance = rf.estimators_[i].feature_importances_
    feat_indices = np.argsort(feat_importance)[::-1]

    # Only plot the top 10 most important features
    feat_indices = feat_indices[0:10]

    print(features_train.columns[feat_indices])

    # plt.figure()
    # plt.title('Random Forest Feature Importances; Class: '+class_to_num[i])
    # plt.bar(range(10), feat_importance[feat_indices],
    #         color="m")
    # plt.xticks(range(10), features_train.columns[feat_indices], rotation=0)
    # plt.xlim([-1, 10])
    # plt.show()