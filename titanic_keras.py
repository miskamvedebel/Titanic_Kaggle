# -*- coding: utf-8 -*-

from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Input
import numpy as np
import pandas as pd
def prep_submission(y_pred, file_name):
    submission = pd.DataFrame({
            "PassengerID": test["PassengerId"],
            "Survived": y_pred[:, 0]})
    submission.to_csv('C:\\Users\\maksim.lebedev\\Desktop\\MachineLearning\\Kaggle\\Titanic\\'+file_name,
                      index = False)
X = np.array(X_train)
y = np.array(y_train.to_frame())


model = Sequential()
#Input layer:
model.add(Dense(units = 5, activation = "relu", input_shape = (17,)))
model.add(Dense(units = 5, activation= "relu"))
model.add(Dense(units = 1, activation="sigmoid"))
model.compile(optimizer = "adam", 
              loss = "binary_crossentropy", 
              metrics=["accuracy"])
model.fit(x = X, y = y, epochs = 3000, batch_size=16)
y_pred = model.predict_classes(x = np.array(X_test), batch_size=16)
prep_submission(y_pred, 'submission_nn_keras_851.csv')

