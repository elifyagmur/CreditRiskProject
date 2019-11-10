# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 22:15:46 2019

@author: yagmur.yilmazturk
"""

import pandas as pd
import catboost as cat
from sklearn.preprocessing import LabelEncoder
import os
import pickle
from sklearn.metrics import roc_auc_score


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



Predictors,Df_train, label_train, removedCols, encodedList = TrainPrep(train_set)
Df_test, encodedList_test = TestPrep(test_set)


#FeaturePath = 'C:\\CreditRiskProject\\Train_Features_Imps.csv'
#DataPath = 'C:\\CreditRiskProject'

#####################Training Classifier############################################################################


def catboosttrainer(X,y,features,initparam,modelname,modelpath,docpath,cvfold = 5):
    print ("searching for optimal iteration count...")
    trainpool = cat.Pool(X[features],y)
    cvresult = cat.cv(params= initparam, fold_count=cvfold, pool=trainpool,stratified = True)
    initparam['iterations'] = (len(cvresult)) - (initparam['od_wait']+1)   
    del initparam['od_wait'] 
    del initparam['od_type']
    print ("optimal iteration count is ", initparam['iterations'])
    print ("fitting model...")
    clf = cat.CatBoostClassifier(** initparam)
    clf.fit(trainpool)
    imp = clf.get_feature_importance(trainpool,fstr_type='FeatureImportance')
    dfimp = pd.DataFrame(imp,columns = ['CatBoostImportance'])
    dfimp.insert(0,column='Feature', value=features) 
    dfimp = dfimp.sort_values(['CatBoostImportance','Feature'], ascending= False)
    xlsxpath = os.path.join(docpath,modelname+".xlsx")
    dfimp.to_excel(xlsxpath)
    print ("pickling model...")
    picklepath = os.path.join(modelpath,modelname)
    with open(picklepath,'wb') as fout:
        pickle.dump(clf, fout)
    return cvresult,clf,initparam,dfimp



modelpath = 'C:\\CreditRiskProject'
docpath = 'C:\\CreditRiskProject'
CatBoostParam = { 'iterations': 2000, 'od_type': 'Iter', 'od_wait': 100,'loss_function': 'Logloss','eval_metric': 'AUC', "random_seed" : 123}

cvresult,clf,initparam,dfimp = catboosttrainer(Df_train,label_train,Predictors,CatBoostParam,'CBmodel_big',modelpath,docpath,cvfold = 5)

predictions = clf.predict_proba(Df_test)[:,1]


#READING SAMPLE SUBMISSION FILE

sample = pd.read_csv(directory+'\\sample_submission.csv')
sample['TARGET']=predictions
#CREATING SUMBISSION FILE
sample.to_csv(directory+ '\\submission_yagmur_rigo2.csv',index=False)

























model = pd.read_pickle(directory+ '\\CBmodel')


proba = model.predict_proba(X_test[Predictors])[:,1]
auc = roc_auc_score(y_test,proba)
print(auc)



