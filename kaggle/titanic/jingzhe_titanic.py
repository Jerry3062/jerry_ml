# 知乎 惊蛰 的专栏分享
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('train.csv', dtype={'Age': np.float64})
test = pd.read_csv('test.csv', dtype={'Age': np.float64})
PassengerId = test['PassengerId']
# 合并数据
all_data = pd.concat([train, test], ignore_index=True)
# print(all_data)
# 展示Sex和存活率的柱状图
# sns.barplot(x='Sex', y='Survived', data=train, palette='Set3')
# plt.show()
# print("Percentage of females who survived:%.2f" % (
#         train["Survived"][train["Sex"] == 'female'].value_counts(normalize=True)[1] * 100))
# print("Percentage of males who survived:%.2f" % (
#         train["Survived"][train["Sex"] == 'male'].value_counts(normalize=True)[1] * 100))
# 展示Pclass和存活率的柱状图
# sns.barplot(x='Pclass',y='Survived',data=train,palette='Set3')
# plt.show()
# survived_data = train['Survived'][train['Survived'] == 1].value_counts()
# print(survived_data)
# sns.barplot(x='SibSp', y='Survived', data=train, palette='Set3')
# plt.show()
# sns.barplot(x='Parch',y='Survived',data=train,palette='Set3')
# plt.show()
facet = sns.FacetGrid(train,hue='Survived',aspect=2)

