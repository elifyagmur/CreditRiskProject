# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:13:15 2019

@author: yagmur.yilmazturk
"""

import pandas as pd
import numpy as np

directory = 'C:\\CreditRiskProject\\Datasets'
train_set = pd.read_csv(directory+ '\\application_train.csv')
test_set = pd.read_csv(directory+ '\\application_test.csv')
bureau = pd.read_csv(directory+ '\\bureau.csv')
pos_cash = pd.read_csv(directory+ '\\POS_CASH_balance.csv')
previous_applicaton = pd.read_csv(directory+ '\\previous_application.csv')
credit_card_balance = pd.read_csv(directory+ '\\credit_card_balance.csv')
installments_payments = pd.read_csv(directory+ '\\installments_payments.csv')

#Feature Generation From Application Data (main data source):
#If a customer has greater income than the credit he applies
train_set['INCOME_GT_CREDIT_FLAG'] = np.where(train_set['AMT_INCOME_TOTAL'] > train_set['AMT_CREDIT'], 1, 0)  #takes 1 if income is greater, 0 o/w

#Credit Income Ratio
train_set['CREDIT_INCOME_PERCENT'] = train_set['AMT_CREDIT'] / train_set['AMT_INCOME_TOTAL']

#Annuity Income Ratio
train_set['ANNUITY_INCOME_PERCENT'] = train_set['AMT_ANNUITY'] / train_set['AMT_INCOME_TOTAL']

# Column to represent Credit Term
train_set['CREDIT_TERM'] = train_set['AMT_CREDIT'] / train_set['AMT_ANNUITY']

# Column to represent Days Employed percent in his life
train_set['DAYS_EMPLOYED_PERCENT'] = train_set['DAYS_EMPLOYED'] / train_set['DAYS_BIRTH']

#Credit-Price of Goods Ratio
train_set['CREDIT_PRICE_OF_GOODS_RATIO'] = train_set['AMT_CREDIT'] / train_set['AMT_GOODS_PRICE']


#Feature Generation From Other Data (other data sources):
##----------------------------------------------------------------------------------------------------------------
#---------Combining Breau Data to Train set----
#group data by id: We can extract number of previous credit, number of active credit, total credit, total debt and total limit for each customer
bureau['CREDIT_ACTIVE_FLAG'] = np.where(bureau['CREDIT_ACTIVE'] == 'Active', 1, 0) #takes 1 if credit is active, 0 o/w
BreauSumTable = bureau[['SK_ID_CURR','CREDIT_ACTIVE_FLAG','AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT','AMT_CREDIT_SUM_LIMIT']].groupby(['SK_ID_CURR']).sum().reset_index()
BreauSumTable = BreauSumTable.rename(columns={'CREDIT_ACTIVE_FLAG':'NUMBER_OF_ACTIVE_CREDIT_BUREAU','AMT_CREDIT_SUM':'TOTAL_CUSTOMER_CREDIT_BUREAU','AMT_CREDIT_SUM_DEBT':'TOTAL_CUSTOMER_DEBT_BUREAU', 'AMT_CREDIT_SUM_LIMIT':'TOTAL_CUSTOMER_LIMIT_BUREAU'})

BreauCountTable = bureau[['SK_ID_CURR','SK_ID_BUREAU']].groupby(['SK_ID_CURR']).count().reset_index()
BreauCountTable = BreauCountTable.rename(columns={'SK_ID_BUREAU':'NUMBER_OF_PREVIOUS_CREDIT_BUREAU'})

#merge groupped data into train set:
train_set =  train_set.merge(BreauSumTable, on='SK_ID_CURR', how='left')
train_set =  train_set.merge(BreauCountTable, on='SK_ID_CURR', how='left')

#fill ne values with zero
train_set.update(train_set[BreauSumTable.columns].fillna(0))
train_set.update(train_set[BreauCountTable.columns].fillna(0))
#train_set.head()


##----------------------------------------------------------------------------------------------------------------
#---------Combining pos_cash Data to Train set----
#group data by id: We can extract number of previous credit, number of active credit, and average of installments for each customer
pos_cash['ACTIVE_STATUS'] = np.where(pos_cash['NAME_CONTRACT_STATUS'] == 'Active', 1, 0) #takes 1 if  active, 0 o/w
pos_cashSum = pos_cash[['SK_ID_CURR','ACTIVE_STATUS']].groupby(['SK_ID_CURR']).sum().reset_index()
pos_cashSum = pos_cashSum.rename(columns={'ACTIVE_STATUS':'NUMBER_OF_ACTIVE_CONTRACT_CASH'})

pos_cashCount = pos_cash[['SK_ID_CURR','SK_ID_PREV']].groupby(['SK_ID_CURR']).count().reset_index()
pos_cashCount = pos_cashCount.rename(columns={'SK_ID_PREV':'NUMBER_OF_PREVIOUS_CREDIT_CASH'})

pos_cashAvg = pos_cash[['SK_ID_CURR','CNT_INSTALMENT','CNT_INSTALMENT_FUTURE']].groupby(['SK_ID_CURR']).mean().reset_index()
pos_cashAvg = pos_cashAvg .rename(columns={'CNT_INSTALMENT':'AVG_OF_INSTALLMENTS_CASH', 'CNT_INSTALMENT_FUTURE':'AVG_OF_INSTALLMENTS_LEFT_CASH'})

#merge groupped data into train set:
train_set =  train_set.merge(pos_cashCount, on='SK_ID_CURR', how='left')
train_set =  train_set.merge(pos_cashSum, on='SK_ID_CURR', how='left')
train_set =  train_set.merge(pos_cashAvg, on='SK_ID_CURR', how='left')

#fill ne values with zero
train_set.update(train_set[pos_cashCount.columns].fillna(0))
train_set.update(train_set[pos_cashSum.columns].fillna(0))
train_set.update(train_set[pos_cashAvg.columns].fillna(0))
#train_set.head()


##----------------------------------------------------------------------------------------------------------------
#---------Combining credit_card_balance Data to Train set----

#group data by id: We can extract number of avtive card contracts, number of previous credit card,
#//total balance, total limit, total drawings amounts and count of drawings of each customer
credit_card_balance['ACTIVE_STATUS'] = np.where(credit_card_balance['NAME_CONTRACT_STATUS'] == 'Active', 1, 0) #takes 1 if  active, 0 o/w
credit_card_balanceSum = credit_card_balance[['SK_ID_CURR','ACTIVE_STATUS']].groupby(['SK_ID_CURR']).sum().reset_index()
credit_card_balanceSum = credit_card_balanceSum.rename(columns={'ACTIVE_STATUS':'NUMBER_OF_ACTIVE_CONTRACT_CARD'})

credit_card_balanceCount = credit_card_balance[['SK_ID_CURR','SK_ID_PREV']].groupby(['SK_ID_CURR']).count().reset_index()
credit_card_balanceCount = credit_card_balanceCount.rename(columns={'SK_ID_PREV':'NUMBER_OF_PREVIOUS_CARD'})

credit_card_balanceTotal = credit_card_balance[['SK_ID_CURR','AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL','AMT_DRAWINGS_CURRENT','CNT_DRAWINGS_CURRENT' ]].groupby(['SK_ID_CURR']).sum().reset_index()
credit_card_balanceTotal = credit_card_balanceTotal .rename(columns={'AMT_BALANCE':'TOTAL_AMOUNT_BALANCE_CARD', 'AMT_CREDIT_LIMIT_ACTUAL':'TOTAL_LIMIT_CARD', 'AMT_DRAWINGS_CURRENT':'TOTAL_DRAVINGS_CARD', 'CNT_DRAWINGS_CURRENT':'COUNT_OF_DRAVINGS_CARD'})

#merge groupped data into train set:
train_set =  train_set.merge(credit_card_balanceSum, on='SK_ID_CURR', how='left')
train_set =  train_set.merge(credit_card_balanceCount, on='SK_ID_CURR', how='left')
train_set =  train_set.merge(credit_card_balanceTotal, on='SK_ID_CURR', how='left')

#fill ne values with zero
train_set.update(train_set[pos_cashCount.columns].fillna(0))
train_set.update(train_set[pos_cashSum.columns].fillna(0))
train_set.update(train_set[pos_cashAvg.columns].fillna(0))
#train_set.head()


##----------------------------------------------------------------------------------------------------------------
#----------Combining previous_applicaton Data to Train set

#group data by id: We can extract count of previous application, count of approved apllications
#//total annuity, total application, total credit of each customer
previous_applicaton['ACTIVE_STATUS'] = np.where(previous_applicaton['NAME_CONTRACT_STATUS'] == 'Approved', 1, 0) #takes 1 if  active, 0 o/w
previous_applicatonSum = previous_applicaton[['SK_ID_CURR','ACTIVE_STATUS']].groupby(['SK_ID_CURR']).sum().reset_index()
previous_applicatonSum = previous_applicatonSum.rename(columns={'ACTIVE_STATUS':'COUNT_OF_APPROVED_APPLS'})

previous_applicatonCount = previous_applicaton[['SK_ID_CURR','SK_ID_PREV']].groupby(['SK_ID_CURR']).count().reset_index()
previous_applicatonCount = previous_applicatonCount.rename(columns={'SK_ID_PREV':'NUMBER_OF_PREVIOUS_APPLS'})

previous_applicatonTotal = previous_applicaton[['SK_ID_CURR','AMT_ANNUITY', 'AMT_APPLICATION','AMT_CREDIT']].groupby(['SK_ID_CURR']).sum().reset_index()
previous_applicatonTotal = previous_applicatonTotal .rename(columns={'AMT_ANNUITY':'TOTAL_ANNUITY_APPLS', 'AMT_APPLICATION':'TOTAL_APPLS', 'AMT_CREDIT':'TOTAL_CREDIT_APPLS'})

#merge groupped data into train set:
train_set =  train_set.merge(previous_applicatonSum, on='SK_ID_CURR', how='left')
train_set =  train_set.merge(previous_applicatonCount, on='SK_ID_CURR', how='left')
train_set =  train_set.merge(previous_applicatonTotal, on='SK_ID_CURR', how='left')

#fill ne values with zero
train_set.update(train_set[previous_applicatonCount.columns].fillna(0))
train_set.update(train_set[previous_applicatonSum.columns].fillna(0))
train_set.update(train_set[previous_applicatonTotal.columns].fillna(0))
#train_set.head()


##----------------------------------------------------------------------------------------------------------------
#----------Combining installments_payments Data to Train set

#group data by id: We can extract average number of installments that each customer makes payment
installments_paymentsAvg = installments_payments[['SK_ID_CURR','NUM_INSTALMENT_NUMBER']].groupby(['SK_ID_CURR']).mean().reset_index()
installments_paymentsAvg = installments_paymentsAvg.rename(columns={'NUM_INSTALMENT_NUMBER':'AVERAGE_OF_INSTALLMENT_PAYMENT'})

#merge groupped data into train set:
train_set =  train_set.merge(installments_paymentsAvg, on='SK_ID_CURR', how='left')

#fill ne values with zero
train_set.update(train_set[installments_paymentsAvg.columns].fillna(0))


#----------Feature Elimination:

#Removing Columns Having Null Valio of ratio more than 0.1
def rmissingvaluecol(dff,threshold):
    l = []
    l = list(dff.drop(dff.loc[:,list((100*(dff.isnull().sum()/len(dff.index))>=threshold))].columns, 1).columns.values)
    print("# Columns having more than %s percent missing values:"%threshold,(dff.shape[1] - len(l)))
    print("Columns:\n",list(set(list((dff.columns.values))) - set(l)))
    return l


remaningColums= rmissingvaluecol(train_set,10) #Here threshold is 10% which means we are going to drop columns having more than 10% of missing values
train_set2 = train_set[remaningColums]

train_set2.to_csv(directory+ '\\train_set_final.csv')



##----------------------------------------------------------------------------------------------------------------
##----------------------------------------------------------------------------------------------------------------
###------------Test Set Preparation (with the columns from test set):

test_set['INCOME_GT_CREDIT_FLAG'] = np.where(test_set['AMT_INCOME_TOTAL'] > test_set['AMT_CREDIT'], 1, 0)  #takes 1 if income is greater, 0 o/w

#Credit Income Ratio
test_set['CREDIT_INCOME_PERCENT'] = test_set['AMT_CREDIT'] / test_set['AMT_INCOME_TOTAL']

#Annuity Income Ratio
test_set['ANNUITY_INCOME_PERCENT'] = test_set['AMT_ANNUITY'] / test_set['AMT_INCOME_TOTAL']

# Column to represent Credit Term
test_set['CREDIT_TERM'] = test_set['AMT_CREDIT'] / test_set['AMT_ANNUITY']

# Column to represent Days Employed percent in his life
test_set['DAYS_EMPLOYED_PERCENT'] = test_set['DAYS_EMPLOYED'] / test_set['DAYS_BIRTH']

#Credit-Price of Goods Ratio
test_set['CREDIT_PRICE_OF_GOODS_RATIO'] = test_set['AMT_CREDIT'] / test_set['AMT_GOODS_PRICE']

##--------------------------
#merge groupped data into train set:
test_set =  test_set.merge(BreauSumTable, on='SK_ID_CURR', how='left')
test_set =  test_set.merge(BreauCountTable, on='SK_ID_CURR', how='left')

#fill ne values with zero
test_set.update(test_set[BreauSumTable.columns].fillna(0))
test_set.update(test_set[BreauCountTable.columns].fillna(0))
#test_set.head()

#merge groupped data into train set:
test_set =  test_set.merge(pos_cashCount, on='SK_ID_CURR', how='left')
test_set =  test_set.merge(pos_cashSum, on='SK_ID_CURR', how='left')
test_set =  test_set.merge(pos_cashAvg, on='SK_ID_CURR', how='left')

#fill ne values with zero
test_set.update(test_set[pos_cashCount.columns].fillna(0))
test_set.update(test_set[pos_cashSum.columns].fillna(0))
test_set.update(test_set[pos_cashAvg.columns].fillna(0))
#test_set.head()

#merge groupped data into train set:
test_set =  test_set.merge(credit_card_balanceSum, on='SK_ID_CURR', how='left')
test_set =  test_set.merge(credit_card_balanceCount, on='SK_ID_CURR', how='left')
test_set =  test_set.merge(credit_card_balanceTotal, on='SK_ID_CURR', how='left')

#fill ne values with zero
test_set.update(test_set[pos_cashCount.columns].fillna(0))
test_set.update(test_set[pos_cashSum.columns].fillna(0))
test_set.update(test_set[pos_cashAvg.columns].fillna(0))
#test_set.head()

#merge groupped data into train set:
test_set =  test_set.merge(previous_applicatonSum, on='SK_ID_CURR', how='left')
test_set =  test_set.merge(previous_applicatonCount, on='SK_ID_CURR', how='left')
test_set =  test_set.merge(previous_applicatonTotal, on='SK_ID_CURR', how='left')

#fill ne values with zero
test_set.update(test_set[previous_applicatonCount.columns].fillna(0))
test_set.update(test_set[previous_applicatonSum.columns].fillna(0))
test_set.update(test_set[previous_applicatonTotal.columns].fillna(0))
#test_set.head()

#merge groupped data into train set:
test_set =  test_set.merge(installments_paymentsAvg, on='SK_ID_CURR', how='left')

#fill ne values with zero
test_set.update(test_set[installments_paymentsAvg.columns].fillna(0))


#Removes TARGET from remaining colums
del remaningColums[1]
test_set2 = test_set[remaningColums]
test_set2.to_csv(directory+ '\\test_set_final.csv')























