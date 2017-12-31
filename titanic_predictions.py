# -*- coding: utf-8 -*-
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
#
X_t, X_s, y_t, y_s = train_test_split(X_train, y_train, test_size = 0.25)
#Random Forest classifier
c_RT = RandomForestClassifier(n_estimators = 100, criterion='entropy')
y_p = c_RT.fit(X_t, y_t).predict(X_s)
f1_score(y_s, y_p)
accuracy_score(y_s, y_p)
#Logistic Regression
c_LR = LogisticRegression()
y_p = c_LR.fit(X_t, y_t).predict(X_s)
f1_score(y_s, y_p)
#Support Vector Machine
c_SVC = LinearSVC()
y_p = c_SVC.fit(X_t, y_t).predict(X_s)
f1_score(y_s, y_p)
#KNN
c_KNN = KNeighborsClassifier(n_neighbors = 10, metric='minkowski')
y_p = c_KNN.fit(X_t, y_t).predict(X_s)
f1_score(y_s, y_p)
#Decision Tree
c_DT = DecisionTreeClassifier()
y_p = c_DT.fit(X_t, y_t).predict(X_s)
f1_score(y_s, y_p)
accuracy_score(y_s, y_p)
#Naive Bayes
c_GNB = GaussianNB()
y_p = c_GNB.fit(X_t, y_t).predict(X_s)
f1_score(y_s, y_p)
y_predict = c_DT.fit(X_train, y_train).predict(X_test)

#Submission
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_predict
    })
submission.to_csv('C:\\Users\\maksim.lebedev\\Desktop\\MachineLearning\\Kaggle\\Titanic\\submission8.csv', 
                  index = False)