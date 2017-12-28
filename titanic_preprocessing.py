# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
train = pd.read_csv('C:\\Users\\maksim.lebedev\\Desktop\\MachineLearning\\Kaggle\\Titanic\\train.csv')
test = pd.read_csv('C:\\Users\\maksim.lebedev\\Desktop\\MachineLearning\\Kaggle\\Titanic\\test.csv')
train_test_data = [train, test]
for ds in train_test_data:
    ds['Title'] = ds.Name.str.extract(' ([A-Za-z]+)\.')
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', \
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}   
for ds in train_test_data:
    ds['Title'] = ds['Title'].map(title_mapping)
    ds['Title'] = ds['Title'].fillna(0)    
    ds['Sex'] = ds['Sex'].map({"male": 0, "female": 1}).astype(int)
    ds['Embarked'] = ds['Embarked'].fillna('S')
    #ds['Embarked'] = ds['Embarked'].map({"S":0, "C":1, "Q":2}).astype(int)
train = pd.get_dummies(train, columns = ['Embarked','Title', 'Pclass'], prefix = ('Embarked', 'Title','Pclass'))
test = pd.get_dummies(test, columns = ['Embarked', 'Title', 'Pclass'], prefix = ('Embarked', 'Title','Pclass'))

train_test_data = [test, train]
for ds in train_test_data:
    avg_age = ds['Age'].mean()
    std_age = ds['Age'].std()
    age_null_count = ds['Age'].isnull().sum()
    random_list = np.random.randint(avg_age-std_age, avg_age+std_age, size = age_null_count)
    ds['Age'][np.isnan(ds['Age'])] = random_list
    ds['Age'] = ds['Age'].astype(int)

for ds in train_test_data:
    ds.loc[ds['Age'] <= 16, 'Age'] = 0
    ds.loc[(ds['Age']>16) & (ds['Age'] <= 32), 'Age'] = 1
    ds.loc[(ds['Age']>32) & (ds['Age'] <= 48), 'Age'] = 2
    ds.loc[(ds['Age']>48) & (ds['Age'] <= 64), 'Age'] = 3
    ds.loc[ds['Age']>64, 'Age'] = 4
for ds in train_test_data:
    ds['Fare'] = ds['Fare'].fillna(ds['Fare'].median())
test['Fare'] = test['Fare'].fillna(train['Fare'].median())
train_test_data = [train, test] #
for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
for ds in train_test_data:
    ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1
for ds in train_test_data:
    ds['IsAlone'] = 0
    ds.loc[ds['FamilySize'] == 1, 'IsAlone'] = 1
#Name Length
train_test_data = [train, test]
for ds in train_test_data:
    ds['NameLen'] = 0
    for i in range(0,len(ds['Name'])):
        ds['NameLen'][i] = len(ds['Name'][i])
for ds in train_test_data:
    ds.loc[ds['NameLen']<=20, 'NameLen'] = 0
    ds.loc[(ds['NameLen']>20)&(ds['NameLen']<=25), 'NameLen'] = 1
    ds.loc[(ds['NameLen']>25)&(ds['NameLen']<=30), 'NameLen'] = 2
    ds.loc[ds['NameLen']>30, 'NameLen'] = 3
features_to_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']
train = train.drop(features_to_drop, axis = 1)
train = train.drop(['PassengerId'], axis = 1)
test = test.drop(features_to_drop, axis = 1)
X_train = train.drop(['Survived'], axis = 1)
y_train = train['Survived']
X_test = test.drop(['PassengerId'], axis = 1).copy()
del(age_null_count, avg_age, dataset, ds, features_to_drop, random_list,
    std_age, title_mapping, i)





