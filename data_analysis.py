import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2

data_path = 'Data/framingham.csv'
raw_data = pd.read_csv(data_path)
raw_data.dropna(inplace=True)

features = raw_data.drop(['TenYearCHD'], axis=1)
targets = raw_data.TenYearCHD

A = raw_data.corr()

print(A.TenYearCHD)

plt.figure()
sns.heatmap(A)
plt.savefig('heatmap.jpg')
