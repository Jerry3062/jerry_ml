import pandas as pd
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv("train.csv")
age_df = train[['Age', 'Pclass', 'Sex']]
age_df = pd.get_dummies(age_df)
known_age = age_df[age_df.Age.notnull()].as_matrix()
unknown_age = age_df[age_df.Age.isnull()].as_matrix()
y = known_age[:, 0]
x = known_age[:, 1:]
rfr = RandomForestRegressor(random_state=0, n_estimators=100)
rfr.fit(x, y)
print(unknown_age)
predictedAges = rfr.predict(unknown_age[:, 1::])
print(predictedAges)
train.loc[(train.Age.isnull()),'Age'] = predictedAges
print(train)

