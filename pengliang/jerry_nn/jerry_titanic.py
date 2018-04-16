import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('F:/dataset/titanic_data/train.csv', dtype={'Age': np.float32})
test = pd.read_csv('F:/dataset/titanic_data/test.csv', dtype={'Age': np.float32})
PassengerId = test['PassengerId']
all_data = pd.concat([train, test], ignore_index=True)
all_data['Title'] = all_data['Name'].apply(lambda x: x.split(",")[1].split(".")[0].strip())
Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
all_data['Title'] = all_data['Title'].map(Title_Dict)

age_df = all_data[['Age', 'Pclass', 'Title']]
age_df = pd.get_dummies(age_df)
known_age = age_df[age_df.Age.notnull()].as_matrix()
unknown_age = age_df[age_df.Age.isnull()].as_matrix()
x = known_age[:, 1:]
y = known_age[:, 0]

rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(x, y)
predictAges = rfr.predict(unknown_age[:, 1:])
all_data.loc[(all_data['Age'].isnull()), 'Age'] = predictAges
all_data.describe()
all_data['FamilyCount'] = all_data['SibSp'] + all_data['Parch'] + 1
train = all_data.loc[(all_data['Survived'].notnull())]


def fam_label(count):
    if count >= 2 and count <= 4:
        return 2
    elif (count > 4) & (count <= 7) | (count == 1):
        return 1
    else:
        return 0


all_data['FamilyLabel'] = all_data['FamilyCount'].apply(fam_label)
train = all_data.loc[(all_data['Survived'].notnull())]
sns.barplot('FamilyLabel', 'Survived', data=train)

all_data = all_data.drop(['Parch', 'SibSp'], axis=1)

all_data['FareBand'] = pd.qcut(all_data['Fare'], 8)
train = all_data.loc[(all_data['Survived'].notnull())]


def fare_label(fare):
    if fare <= 10:
        return 0
    elif fare <= 69:
        return 1
    else:
        return 2


all_data['FareLabel'] = all_data['Fare'].apply(fare_label)

all_data = all_data.drop(['Fare'], axis=1)
train = all_data.loc[(all_data['Survived'].notnull())]

all_data['Embarked'].fillna('S')
TicketCount = dict(all_data['Ticket'].value_counts())
all_data['TicketCount'] = all_data['Ticket'].apply(lambda x: TicketCount[x])
train = all_data.loc[(all_data['Survived'].notnull())]


def ticket_label(count):
    if (count >= 2) & (count <= 4):
        return 2
    elif ((count > 4) & (count <= 8)) | (count == 1):
        return 1
    elif (count > 8):
        return 0


all_data['TicketLabel'] = all_data['TicketCount'].apply(ticket_label)
all_data = all_data.drop(['Ticket', 'TicketCount'], axis=1)
all_data['AgeBand'] = pd.cut(all_data['Age'], 6)
train = all_data.loc[(all_data['Survived'].notnull())]


def age_band(age):
    if age <= 10:
        return 0
    elif age <= 60:
        return 1
    else:
        return 2


all_data['AgeLabel'] = all_data['Age'].apply(age_band)
all_data = all_data.drop(['Age', 'AgeBand'], axis=1)
all_data['Embarked'] = all_data['Embarked'].fillna('S')

all_data = all_data.drop(['Cabin', 'FareBand', ], axis=1)
all_data = all_data[['Survived', 'Pclass', 'Sex', 'AgeLabel', 'FareLabel', 'Embarked', 'Title',
                     'FamilyLabel', 'TicketLabel']]


def norm(i):
    if i == 0:
        return 'A'
    elif i == 1:
        return 'B'
    elif i == 2:
        return 'C'
    elif i == 3:
        return 'D'
    elif i == 4:
        return 'E'


all_data['Pclass'] = all_data['Pclass'].apply(norm)
all_data['AgeLabel'] = all_data['AgeLabel'].apply(norm)
all_data['FareLabel'] = all_data['FareLabel'].apply(norm)
all_data['FamilyLabel'] = all_data['FamilyLabel'].apply(norm)
all_data['TicketLabel'] = all_data['TicketLabel'].apply(norm)
all_data = pd.get_dummies(all_data)

train = all_data[all_data['Survived'].notnull()]
test = all_data[all_data['Survived'].isnull()].drop('Survived', axis=1)
x = train.as_matrix()[:, 1:]
y = train.as_matrix()[:, 0]

print(x.shape)

x_test = test.as_matrix()
print(x_test.shape)

# pipe = Pipeline([('select', SelectKBest(k=16)), ('classify', RandomForestClassifier(random_state=10,
#                                                                                     max_features='sqrt'))])
# param_test = {'classify__n_estimators': list(range(20, 50, 2)),
#               'classify__max_depth': list(range(2, 30, 2))}
# gsearch = GridSearchCV(estimator=pipe, param_grid=param_test, scoring='roc_auc', cv=10)
# gsearch.fit(x, y)
# print(gsearch.best_params_, gsearch.best_score_)
# 2)训练模型
# select = SelectKBest(k=16)
# clf = RandomForestClassifier(random_state=10, warm_start=True,
#                              n_estimators=28,
#                              max_depth=6,
#                              max_features='sqrt')
# pipeline = make_pipeline(select, clf)
# pipeline.fit(x, y)
# # 3)交叉验证
# cv_score = cross_validation.cross_val_score(pipeline, x, y, cv=10)
# print('CV Score:mean-%.7g | Std - %.7g ' % (np.mean(cv_score), np.std(cv_score)))
#
# predictions = pipeline.predict(test)
# submission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions.astype(np.int32)})
# submission.to_csv("submission.csv", index=False)

# Decision Tree

# decision_tree = RandomForestClassifier(n_estimators=10)
# decision_tree.fit(x, y)
# Y_pred = decision_tree.predict(test.as_matrix())
# acc_decision_tree = round(decision_tree.score(x, y) * 100, 2)
# submission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': Y_pred.astype(np.int32)})
# submission.to_csv("submission.csv", index=False)

from pengliang.jerry_nn.neural_network import NeuralNetwork

nn = NeuralNetwork([26, 100, 2])
nn.fit(x, y)
predictions = []
for i in range(x_test.shape[0]):
    o = nn.predict(x_test[i])
    predictions.append(np.argmax(o))
submission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
submission.to_csv("F:/dataset/titanic_data/submission.csv", index=False)
