import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv', dtype={'Age': np.float64})
test = pd.read_csv('test.csv', dtype={'Age': np.float64})
PassengerId = test['PassengerId']
all_data = pd.concat([train, test], ignore_index=True)
# sns.barplot('SibSp','Survived',data=train,palette='Set3')
# plt.show()
# facet = sns.FacetGrid(train, hue='Survived', aspect=2)
# facet.map(sns.kdeplot,'Fare',shade=True)
# facet.set(xlim=(0,200))
# facet.add_legend()
# plt.show()

all_data['Title'] = all_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
all_data['Title'] = all_data['Title'].apply(lambda x: Title_Dict[x])

# sns.barplot(x="Title", y="Survived", data=all_data, palette='Set3')
# plt.show()
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1


# sns.barplot('FamilySize','Survived',data=all_data)
# plt.show()

def fam_labes(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0


all_data['FamilyLabel'] = all_data['FamilySize'].apply(fam_labes)
# sns.barplot('FamilyLabel', 'Survived', data=all_data)
# plt.show()
all_data['Cabin'] = all_data['Cabin'].fillna('Unknown')
all_data['Deck'] = all_data['Cabin'].str[0]
# sns.barplot('Deck', 'Survived', data=all_data)
# plt.show()
Ticket_Count = dict(all_data['Ticket'].value_counts())
# print(Ticket_Count)
all_data['TicketGroup'] = all_data['Ticket'].apply(lambda x: Ticket_Count[x])


# sns.barplot('TicketGroup', 'Survived', data=all_data)
# plt.show()

def Ticket_Label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0


all_data['TicketLabel'] = all_data['TicketGroup'].apply(Ticket_Label)
age_df = all_data[['Age', 'Pclass', 'Title']]
age_df = pd.get_dummies(age_df)
known_age = age_df[age_df.Age.notnull()].as_matrix()
unknown_age = age_df[age_df.Age.isnull()].as_matrix()
x = known_age[:, 1:]
y = known_age[:, 0]
rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(x, y)
predictAges = rfr.predict(unknown_age[:, 1:])
all_data.loc[(all_data.Age.isnull()), 'Age'] = predictAges
# sns.boxplot('Embarked','Fare',hue='Pclass',data=all_data)
# plt.show()
all_data['Embarked'] = all_data['Embarked'].fillna('C')
fare = all_data[(all_data['Embarked'] == 'S') & (all_data['Pclass'] == 3)].Fare.median()
all_data['Fare'] = all_data['Fare'].fillna(fare)

all_data = all_data[
    ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilyLabel', 'Deck', 'TicketLabel']]
all_data = pd.get_dummies(all_data)
train = all_data[all_data['Survived'].notnull()]
test = all_data[all_data['Survived'].isnull()].drop('Survived', axis=1)
x = train.as_matrix()[:, 1:]
y = train.as_matrix()[:, 0]

# pipe = Pipeline(
#     [('select', SelectKBest(k=18)), ('classify', AdaBoostClassifier(random_state=10, ))])
# param_test = {'classify__n_estimators': list(range(20, 50, 2)),
#               'classify__max_depth': list(range(2, 40, 2))}
# gsearch = GridSearchCV(estimator=pipe, param_grid=param_test, scoring='roc_auc', cv=8)
# gsearch.fit(x, y)
# print(gsearch.best_params_, gsearch.best_score_)

select = SelectKBest(k=18)
clf = GradientBoostingClassifier(random_state=10, max_depth=6, max_features='sqrt',
                                 warm_start=True,
                                 n_estimators=28)
pipeline = make_pipeline(select, clf)
pipeline.fit(x, y)
cv_score = cross_validation.cross_val_score(pipeline, x, y, cv=8)
print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))
predictions = pipeline.predict(test)
submission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions.astype(np.int32)})
submission.to_csv('jerry_adaboost_submission.csv', index=False)
