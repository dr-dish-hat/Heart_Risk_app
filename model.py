import os
import pandas as pd
import numpy as np
import imblearn
from imblearn.over_sampling import SMOTE
import sklearn as sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
import pickle

'''
Selected Features

sysBP -> numerical
glucose -> numerical
age -> numerical
totChol -> numerical
cigsPerDay -> numerical
diaBP -> numerical
prevalentHyp -> categorical binary
diabetese -> categorical binary
'''

# Reading and processing data

data_path = 'Data/framingham.csv'
raw_data = pd.read_csv(data_path)
raw_data.dropna(inplace=True)

feature_cols = ['sysBP', 'diaBP',  'age', 'glucose', 'totChol', 'cigsPerDay',
                'prevalentHyp', 'diabetes']

features = raw_data[feature_cols]
targets = raw_data.TenYearCHD

print(features.head())
# Minority over sampling

sm = SMOTE()
features, targets = sm.fit_resample(features, targets)

model_data = tuple()
model_data = train_test_split(features, targets, test_size=0.2, random_state=24)

features_train = model_data[0]
features_test = model_data[1]
targets_train = model_data[2]
targets_test = model_data[3]

# Model

RFC = RandomForestClassifier(n_estimators=512)
RFC.fit(features_train, targets_train)

# Testing model

predict_train = RFC.predict(features_train)
print(f1_score(predict_train, targets_train))

prediction = RFC.predict(features_test)
print('f1_score :', f1_score(prediction, targets_test))
print('confusion matrix :', confusion_matrix(prediction, targets_test))

# Saving model

pickle.dump(RFC, open('model.pkl', 'wb'))
