# 参照知乎大树先生的教程
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# print(train_data.describe())
# train_data['Survived'].value_counts().plot.pie(autopct='%1.2f%%')
# plt.show()
train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().values
train_data['Cabin'] = train_data.Cabin.fillna('UO')
age_df = train_data[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
age_df_notnull = age_df.loc[(train_data['Age'].notnull())]
age_df_isnull = age_df.loc[(train_data['Age'].isnull())]
x = age_df_notnull.values[:, 1:]
y = age_df_notnull.values[:, 0]
rfr = RandomForestRegressor(n_estimators=50)
rfr.fit(x, y)
predictAges = rfr.predict(age_df_isnull.values[:, 1:])
train_data.loc[(train_data.Age.isnull()), 'Age'] = predictAges
train_data[['Sex','Pclass','Survived']].groupby(['Pclass','Sex']).mean().plot.bar()
plt.show()