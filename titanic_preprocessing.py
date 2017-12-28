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
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, accuracy_score
X_t, X_s, y_t, y_s = train_test_split(X_train, y_train, test_size = 0.25)
from sklearn.ensemble import RandomForestClassifier
c_RT = RandomForestClassifier(n_estimators = 100, criterion='entropy')
y_p = c_RT.fit(X_t, y_t).predict(X_s)
f1_score(y_s, y_p)
accuracy_score(y_s, y_p)
from sklearn.linear_model import LogisticRegression
c_LR = LogisticRegression()
y_p = c_LR.fit(X_t, y_t).predict(X_s)
f1_score(y_s, y_p)
from sklearn.svm import SVC, LinearSVC
c_SVC = LinearSVC()
y_p = c_SVC.fit(X_t, y_t).predict(X_s)
f1_score(y_s, y_p)
from sklearn.neighbors import KNeighborsClassifier
c_KNN = KNeighborsClassifier(n_neighbors = 10, metric='minkowski')
y_p = c_KNN.fit(X_t, y_t).predict(X_s)
f1_score(y_s, y_p)
from sklearn.tree import DecisionTreeClassifier
c_DT = DecisionTreeClassifier()
y_p = c_DT.fit(X_t, y_t).predict(X_s)
f1_score(y_s, y_p)
accuracy_score(y_s, y_p)
from sklearn.naive_bayes import GaussianNB
c_GNB = GaussianNB()
y_p = c_GNB.fit(X_t, y_t).predict(X_s)
f1_score(y_s, y_p)
from sklearn.linear_model import SGDClassifier
c_SGD = SGDClassifier(alpha = 0.0005)
y_p = c_SGD.fit(X_t, y_t).predict(X_s)
f1_score(y_s, y_p)
y_predict = c_RT.fit(X_train, y_train).predict(X_test)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_predict
    })
submission.to_csv('C:\\Users\\maksim.lebedev\\Desktop\\MachineLearning\\Kaggle\\Titanic\\submission7.csv', 
                  index = False)

#Playing with data

train = pd.read_csv('C:\\Users\\maksim.lebedev\\Desktop\\MachineLearning\\Kaggle\\Titanic\\train.csv')
test = pd.read_csv('C:\\Users\\maksim.lebedev\\Desktop\\MachineLearning\\Kaggle\\Titanic\\test.csv')
train.head()
train.shape
train.describe(include = ['O'])
train.info()
train.isnull().sum()[train.isnull().sum().values>0]
test.shape
test.info()
test.isnull().sum()[test.isnull().sum().values>0]
survived = train[train["Survived"] == 1]
not_survived = train[train["Survived"] == 0]
'''relationships btw survivals and features'''
print ("Survived: %i (%.1f%%)"%(len(survived), float(len(survived)/len(train)*100.0)))
print ("Not survived: %i (%.1f%%)"%(len(not_survived), float(len(not_survived)/len(train)*100.0)))
print ('Total: %i'%(len(train)))
train["Pclass"].value_counts()
train.groupby('Pclass')["Survived"].value_counts()
train[["Pclass", "Survived"]].groupby(["Pclass"], as_index = 'False').mean()
sns.barplot(x = 'Pclass', y = 'Survived', data = train)
train["Sex"].value_counts()
train[["Sex", "Survived"]].groupby(["Sex"], as_index = 'False').mean()
sns.barplot(x = 'Sex', y = 'Survived', data = train)
tab = pd.crosstab(index = train["Pclass"], columns = train["Sex"] )
tab.div(tab.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True)
plt.xlabel('Pclass')
plt.ylabel('Percentage')
sns.factorplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data = train)

sns.factorplot(x = 'Pclass', y = 'Survived', hue = 'Sex', data = train, col = 'Embarked')
train.Embarked.value_counts()
train.groupby('Embarked').Survived.value_counts()
train[['Embarked', 'Survived']].groupby('Embarked', as_index = 'False').mean()
sns.barplot(x = 'Embarked', y = 'Survived', data = train)

train.Parch.value_counts()
train.groupby('Parch').Survived.value_counts()
train[['Parch', 'Survived']].groupby('Parch').mean()
sns.barplot(x = 'Parch', y = 'Survived', data = train)

train.SibSp.value_counts()
train.groupby('SibSp').Survived.value_counts()
train[['SibSp', 'Survived']].groupby('SibSp').mean()
sns.barplot(x = 'SibSp', y = 'Survived', data = train, ci = None)

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

sns.violinplot(x = 'Embarked', y = 'Age', hue = 'Survived', data = train, split = True, ax = ax1)
sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = train, split = True, ax = ax2)
sns.violinplot(x = 'Sex', y = 'Age', hue = 'Survived', data = train, split = True, ax = ax3)
fig

total_survived = train[train['Survived'] == 1]
total_not_survived = train[train['Survived'] == 0]
male_survived = train[(train['Survived'] == 1) & (train['Sex'] == 'male')]
female_survived = train[(train['Survived'] == 1) & (train['Sex'] == 'female')]
male_not_survived = train[(train['Survived'] == 0) & (train['Sex'] == 'male')]
female_not_survived = train[(train['Survived'] == 0) & (train['Sex'] == 'female')]
plt.figure(figsize=[15,5])
plt.subplot(111)
sns.distplot(total_survived['Age'].dropna().values, 
             bins = range(0, 81, 1), kde = False, color = 'blue')
sns.distplot(total_not_survived['Age'].dropna().values, 
             bins = range(0, 81, 1), kde = False, color = 'red', axlabel="Age")

plt.figure(figsize=[15,5])

plt.subplot(121)
sns.distplot(female_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(female_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Female Age')

plt.subplot(122)
sns.distplot(male_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(male_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Male Age')

plt.figure(figsize = [15,6])
sns.heatmap(train.drop(["PassengerId"], axis = 1).corr(), vmax = 0.6, square=True, annot=True)
train.drop('PassengerId', axis = 1).corr()

train_test_data = [train, test]
for ds in train_test_data:
    ds['Title'] = ds.Name.str.extract(' ([A-Za-z]+)\.')
pd.crosstab(index = train['Title'], columns = train['Sex'])
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', \
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby('Title').mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}   
for ds in train_test_data:
    ds['Title'] = ds['Title'].map(title_mapping)
    ds['Title'] = ds['Title'].fillna(0)
    
for ds in train_test_data:
    ds['Sex'] = ds['Sex'].map({"male": 0, "female": 1}).astype(int)
for ds in train_test_data:
    ds['Embarked'] = ds['Embarked'].fillna('S')
for ds in train_test_data:
    ds['Embarked'] = ds['Embarked'].map({"S":0, "C":1, "Q":2}).astype(int)
train_test_data = [test, train]
for ds in train_test_data:
    avg_age = ds['Age'].mean()
    std_age = ds['Age'].std()
    age_null_count = ds['Age'].isnull().sum()
    random_list = np.random.randint(avg_age-std_age, avg_age+std_age, size = age_null_count)
    ds['Age'][np.isnan(ds['Age'])] = random_list
    ds['Age'] = ds['Age'].astype(int)
train['AgeBand'] = pd.cut(train['Age'], 5)
'''a = train['Age'].mean()
s = train['Age'].std()
train['Age'][np.isnan(train['Age'])] = np.random.randint(a-s, a+s, train['Age'].isnull().sum())
for i in range(0,len(train['Age'])):
    if train['Age'][i] <= 16:
        train['Age'][i] = 0
    elif train['Age'][i] <= 32:
        train['Age'][i] = 1
    elif train['Age'][i] <= 48:
        train['Age'][i] = 2
    elif train['Age'][i] <= 64:
        train['Age'][i] = 3
    else:
        train['Age'][i] = 4'''
print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())
for ds in train_test_data:
    ds.loc[ds['Age'] <= 16, 'Age'] = 0
    ds.loc[(ds['Age']>16) & (ds['Age'] <= 32), 'Age'] = 1
    ds.loc[(ds['Age']>32) & (ds['Age'] <= 48), 'Age'] = 2
    ds.loc[(ds['Age']>48) & (ds['Age'] <= 64), 'Age'] = 3
    ds.loc[ds['Age']>64, 'Age'] = 4
train = train.drop('AgeBand', axis = 1)
for ds in train_test_data:
    ds['Fare'] = ds['Fare'].fillna(ds['Fare'].median())
test['Fare'] = test['Fare'].fillna(train['Fare'].median())
train_test_data = [train, test] '''Merging train and test data again'''
for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
for ds in train_test_data:
    ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1
train[['FamilySize', 'Survived']].groupby('FamilySize').mean()
sns.barplot(x ='FamilySize', y = 'Survived', data = train, ci = None)
for ds in train_test_data:
    ds['IsAlone'] = 0
    ds.loc[ds['FamilySize'] == 1, 'IsAlone'] = 1
train[['IsAlone', 'Survived']].groupby('IsAlone').mean()
features_to_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']
train = train.drop(features_to_drop, axis = 1)
train = train.drop(['PassengerId'], axis = 1)
test = test.drop(features_to_drop, axis = 1)
X_train = train.drop(['Survived'], axis = 1)
y_train = train['Survived']
X_test = test.drop(['PassengerId'], axis = 1).copy()


