import pandas
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

import re
import matplotlib as plt
import jerry_numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier

data = pandas.read_csv("train.csv")
data['age'] = data['age'].fillna(data['age'].median())
data.loc[data['sex'] == 'male', 'sex'] = 0
data.loc[data['sex'] == 'female', 'sex'] = 1
data['embarked'] = data['embarked'].fillna('S')
data.loc[data['embarked'] == 'S', 'embarked'] = 0
data.loc[data['embarked'] == 'C', 'embarked'] = 1
data.loc[data['embarked'] == 'Q', 'embarked'] = 2

# predictors = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']

# random_forests = RandomForestClassifier(random_state=1,
#                                         n_estimators=10,min_samples_split=2,min_samples_leaf=1)
# kf = cross_validation.KFold(data.shape[0],n_folds=3,random_state=1)
# scores = cross_validation.cross_val_score(random_forests,data[predictors],
#                                           data['survived'],cv=kf)
# print(scores.mean())
# random_forests2 = RandomForestClassifier(random_state=1,
#                                          n_estimators=50,min_samples_split=4,min_samples_leaf=2)
# scores=cross_validation.cross_val_score(random_forests2,
#                                         data[predictors],data['survived'],cv=kf)
# print(scores.mean())
data['familysize'] = data['sibsp'] + data['parch']
data['namelength'] = data['name'].apply(lambda x: len(x))

import re


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""


titles = data['name'].apply(get_title)
print(pandas.value_counts(titles))
title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5, 'Rev': 6, 'Major': 7, 'Cos': 7,
                 'Jonkheer': 8, 'Mme': 9, 'Ms': 10, 'Capt': 11, 'Countess': 12,
                 'Sir': 13, 'Don': 14, 'Lady': 15, 'Col': 16, 'Mlle': 17}
for k, v in title_mapping.items():
    titles[titles == k] = v
print(pandas.value_counts(titles))
data['title'] = titles

predictors = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked',
              'familysize', 'title', 'namelength']
selector = SelectKBest(f_classif, k=5)
selector.fit(data[predictors], data['survived'])
scores = -np.log10(selector.pvalues_)
print(scores)

predictors = ['pclass', 'sex', 'fare', 'title', 'namelength']
kf = cross_validation.KFold(data.shape[0], n_folds=3, random_state=1)
random_forests2 = RandomForestClassifier(random_state=1,
                                         n_estimators=50, min_samples_split=4, min_samples_leaf=2)
scores = cross_validation.cross_val_score(random_forests2,
                                          data[predictors], data['survived'], cv=kf)
print(scores.mean())

algorithms = [[GradientBoostingClassifier(random_state=1, n_estimators=25,max_depth=3),
               ['pclass', 'sex', 'fare', 'title', 'namelength']],[LogisticRegression(random_state=1),
                                                                  ['pclass', 'sex', 'fare', 'title', 'namelength']]]
predictions = []
for train, test in kf:
    train_target = data['survived'].iloc[train]
    full_test_predictions = []
    for alg, predictors in algorithms:
        alg.fit(data[predictors].iloc[train, :], train_target)
        test_predictions = alg.predict_proba(data[predictors].iloc[test, :].astype(float))[:, 1]
        full_test_predictions.append(test_predictions)
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)
accuracy = sum(predictions[predictions == data['survived']]) / len(predictions)
print(accuracy)
