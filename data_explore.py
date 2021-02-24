import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_selection import SelectKBest, chi2

data_path = 'Data/framingham.csv'
raw_data = pd.read_csv(data_path)
raw_data = raw_data.dropna()
raw_data.rename(columns={"male":"gender"}, inplace=True)

X = raw_data.drop('TenYearCHD', axis=1)
y = raw_data.TenYearCHD

best = SelectKBest(score_func=chi2, k = 8)

fit = best.fit(X, y)

data_scores = pd.DataFrame(fit.scores_)
data_columns = pd.DataFrame(X.columns)

scores = pd.concat([data_columns, data_scores], axis=1)
scores.columns = ['Feature', 'score']

print(scores.nlargest(10, 'score'))
