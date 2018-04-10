import pandas
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

data = pandas.read_csv("train.csv")
data['age'] = data['age'].fillna(data['age'].median())
data.loc[data['sex'] == 'male', 'sex'] = 0
data.loc[data['sex'] == 'female', 'sex'] = 1
data['embarked'] = data['embarked'].fillna('S')
data.loc[data['embarked'] == 'S', 'embarked'] = 0
data.loc[data['embarked'] == 'C', 'embarked'] = 1
data.loc[data['embarked'] == 'Q', 'embarked'] = 2

predictors = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
linear_regression = LinearRegression()
kf = KFold(data.shape[0], n_folds=3, random_state=1)
predictions = []
for train, test in kf:
    train_predictors = (data[predictors].iloc[train, :])
    train_target = data['survived'].iloc[train]
    linear_regression.fit(train_predictors, train_target)
    test_predictions = linear_regression.predict(data[predictors].iloc[test, :])
    predictions.append(test_predictions)
import jerry_numpy as np

predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 1
accuracy = sum(predictions[predictions == data['survived']]) / len(predictions)
print(accuracy)
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression()
scores = cross_validation.cross_val_score(logistic_regression, data[predictors], data['survived'], cv=3)
print(scores.mean())
