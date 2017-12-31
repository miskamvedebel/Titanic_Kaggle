# -*- coding: utf-8 -*-

#Playing with data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
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

train['Ticket'].value_counts()[train['Ticket'].value_counts()>1]
train['Fare'][train['Ticket'] == '347082']
train[train['Ticket'] == 'CA. 2343']
test['Ticket'].value_counts()[test['Ticket'].value_counts()>1]
test[test['Ticket'] == '29103']
train.isnull().sum()
X_train = train.drop(['Survived'], axis = 1)
X_test = test.copy()

X = pd.concat((X_train, X_test), ignore_index = True)
tickets = X['Ticket'].value_counts()[X['Ticket'].value_counts()>1]
tickets = pd.DataFrame({"Ticket":tickets.index,
                        "Count": tickets.values})
tickets['Fare'] = 0.00
for i in range(0, len(tickets)):
    tickets['Fare'][i] = X['Fare'][X['Ticket'] == tickets['Ticket'][i]].mean()/tickets['Count'][i]
for i in range (0, len(tickets)):
    for j in range (0, len(X)):
        if (str(X['Ticket'][j]) == str(tickets['Ticket'][i])):
            X['Fare'][j] = tickets['Fare'][i]
