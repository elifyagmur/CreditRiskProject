# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 18:58:17 2019

@author: yagmur.yilmazturk
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

directory = 'C:\\CreditRiskProject\\Datasets'
train_set = pd.read_csv(directory+ '\\train_set_final.csv', index_col=0)
test_set = pd.read_csv(directory+ '\\test_set_final.csv', index_col=0)

#Remove Unnecessary colums
def removenonuniquecol(dataset):
    dropcols = [col for col in dataset.columns if dataset[col].nunique(dropna=True)==1]
    print ('Removing columns: ',dropcols)
    dataset.drop(dropcols,axis= 1,inplace= True,errors= 'ignore')
    return dropcols

#For label encoding
def labelencoder(dataset):
    objectlist = dataset.select_dtypes(include=['object']).copy()
    cat_col = [col for col in dataset.columns if col in objectlist]
    for col in cat_col:
        print("Encoding ",col)
        lbl = LabelEncoder()
        dataset[col].fillna(-999)
        lbl.fit(list(dataset[col].values.astype('str')))
        dataset[col] = lbl.transform(list(dataset[col].values.astype('str')))
    return cat_col

#######################Trainset-Testset preperation#####################################################

def TrainPrep(Datasetname):
    removedCols = removenonuniquecol(Datasetname)
    Predictors = [col for col in Datasetname]
    Predictors = [col for col in Predictors if col not in removedCols]
    Predictors = [col for col in Predictors if col not in ['TARGET']]
    DfLabel = Datasetname['TARGET']
    encodedList = labelencoder(Datasetname)
    return Predictors,Datasetname, DfLabel, removedCols, encodedList


def TestPrep(Datasetname):
    encodedList = labelencoder(Datasetname)
    return Datasetname, encodedList

#Prepare train and test sets for the model
Predictors,Df_train, label_train, removedCols, encodedList = TrainPrep(train_set)
Df_test, encodedList_test = TestPrep(test_set)

X_train = Df_train[Predictors]
Y_tarin = label_train
x_test = Df_test

#fill na values with the mean of the column
X_train = X_train.fillna(X_train.mean())

#Fit the RF model:
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train,Y_tarin)

#Predict y_test
x_test = x_test.fillna(x_test.mean())
y_test = regressor.predict(x_test)

#READING SAMPLE SUBMISSION FILE
sample = pd.read_csv(directory+'\\sample_submission.csv')
sample['TARGET']=y_test
#CREATING SUMBISSION FILE
sample.to_csv(directory+ '\\submission_yagmur_rigo2_RandomForest.csv',index=False)

































