# Importing all required libraries
import numpy as np
import pandas as pd
import lightgbm as lgb
from fastai.imports import *
from fastai.structured import *
import matplotlib.pyplot as plt
from imblearn.under_sampling import ClusterCentroids
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# Loading the excel file
df_raw = pd.read_excel('Dataset.xlsx',
sheetname=0,
header=0,
index_col=False,
keep_default_na=True
)

# For plotting undersampled data in 2d sapce
def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()

# Error calculation of model
def error(x,y):
    print(confusion_matrix(x, y))
    print(f1_score(x, y))
    print(classification_report(x, y))

# Model Score 
def print_score(m):
    res = [error(m.predict(X_train), y_train),
           error(m.predict(X_valid), y_valid),
           m.score(X_train, y_train), m.score(X_valid, y_valid)]

# Convert string to categorical type and set order of categories
train_cats(df_raw)
df_raw.Month.cat.set_categories(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], ordered=True, inplace=True)
df_raw.DayOfWeek.cat.set_categories(['Monday', 'Tuesday','Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True, inplace=True)
df_raw.DayOfWeekClaimed.cat.set_categories(['Monday', 'Tuesday','Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True, inplace=True)
df_raw.MonthClaimed.cat.set_categories(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], ordered=True, inplace=True) 
# Convert data in test set
apply_cats(df=df_test, trn=df_raw)

# Convert category to numerical and replace missing data
df, y, nas = proc_df(df_raw, 'FraudFound_P')
X_test,_,nas = proc_df(df_test, na_dict=nas)
df, y , nas = proc_df(df_raw, 'FraudFound_P', na_dict=nas)

# Undersample majority class
cc = ClusterCentroids(ratio={0: 6650}, n_jobs=-1)
X_cc_full, y_cc_full = cc.fit_sample(df, y)
plot_2d_space(X_cc, y_cc, 'Cluster Centroids under-sampling')

# Model(LGBoost)
lgb_train = lgb.Dataset(X_cc_full, y_cc_full, free_raw_data=False)
# Parametes for lgboost
parameters = {'num_leaves': 2**5,
              'learning_rate': 0.05,
              'is_unbalance': True,
              'min_split_gain': 0.03,
              'min_child_weight': 1,
              'reg_lambda': 1,
              'subsample': 1,
              'objective':'binary',
              'task': 'train'
              }
num_rounds = 500
# Model fitting
clf = lgb.train(parameters, lgb_train, num_boost_round=num_rounds)

# Scores attained
y_prob = clf.predict(X_valid)
y_pred = np.array([0 if i<0.45 else 1 for i in y_prob])
error(y_pred, y_valid)  

# Getting test predictions
y_prob = clf.predict(X_test)
y_pred_test = np.array([0 if i<0.45 else 1 for i in y_prob])
y_pred_df = pd.DataFrame(y_pred_test.reshape(-1,1))
y_pred_df.columns = ['FraunFound_P']
y_pred_df.to_csv('test_results.csv', index=False)