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
# 展示Sex和存活率的柱状图  Sex Feature：女性幸存率远高于男性
# sns.barplot(x='Sex', y='Survived', data=train, palette='Set3')
# plt.show()

# print("Percentage of females who survived:%.2f" % (
#         train["Survived"][train["Sex"] == 'female'].value_counts(normalize=True)[1] * 100))
# print("Percentage of males who survived:%.2f" % (
#         train["Survived"][train["Sex"] == 'male'].value_counts(normalize=True)[1] * 100))
# 展示Pclass和存活率的柱状图 Pclass Feature：乘客社会等级越高，幸存率越高
# sns.barplot(x='Pclass',y='Survived',data=train,palette='Set3')
# plt.show()

# survived_data = train['Survived'][train['Survived'] == 1].value_counts()
# print(survived_data)

# SibSp Feature：配偶及兄弟姐妹数适中的乘客幸存率更高
# sns.barplot(x='SibSp', y='Survived', data=train, palette='Set3')
# plt.show()

# Parch Feature：父母与子女数适中的乘客幸存率更高
# sns.barplot(x='Parch',y='Survived',data=train,palette='Set3')
# plt.show()

# Age Feature：未成年人幸存率高于成年人
# facet = sns.FacetGrid(train,hue='Survived',aspect=2)
# facet.map(sns.kdeplot,'Age',shade=True)
# xlim x_limit x轴的最大最小
# facet.set(xlim=(0,train['Age'].max()))
# facet.add_legend()
# plt.show()

# Fare Feature：支出船票费越高幸存率越高
# facet = sns.FacetGrid(train, hue='Survived', aspect=2)
# facet.map(sns.kdeplot, 'Fare', shade=True)
# facet.set(xlim=(0, 200))
# facet.add_legend()
# plt.show()

# Title Feature(New)：不同称呼的乘客幸存率不同
all_data['Title'] = all_data['Name'].apply(lambda x: x.split(',')[1].split(".")[0].strip())
Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
all_data['Title'] = all_data['Title'].map(Title_Dict)
# sns.barplot(x='Title', y='Survived', data=all_data, palette='Set3')
# plt.show()

# FamilyLabel Feature(New)：家庭人数为2到4的乘客幸存率较高
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1


# sns.barplot(x='FamilySize', y='Survived', data=all_data, palette='Set3')
# plt.show()

# 按生存率把FamilySize分为三类，构成FamilyLabel特征。
def fam_label(family_count):
    if (family_count >= 2) & (family_count <= 4):
        return 2
    elif ((family_count > 4) & (family_count <= 7)) | (family_count == 1):
        return 1
    elif family_count > 7:
        return 0


all_data['FamilyLabel'] = all_data['FamilySize'].apply(fam_label)
# sns.barplot(x='FamilyLabel', y='Survived', data=all_data, palette='Set3')
# plt.show()

# Deck Feature(New)：不同甲板的乘客幸存率不同
all_data['Cabin'] = all_data['Cabin'].fillna('Unknown')
all_data['Deck'] = all_data['Cabin'].str[0]
# sns.barplot(x='Deck',y='Survived',data=all_data,palette='Set3')
# plt.show()

# TicketGroup Feature(New)：与2至4人共票号的乘客幸存率较高
Ticket_Count = dict(all_data['Ticket'].value_counts())
# print(all_data['Ticket'].value_counts())
# print(Ticket_Count)
all_data['TicketGroup'] = all_data['Ticket'].apply(lambda x: Ticket_Count[x])


# sns.barplot(x='TicketGroup',y='Survived',data=all_data,palette='Set3')
# plt.show()
# 按生存率把TicketGroup分为三类。
def ticket_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0


all_data['TicketGroup'] = all_data['TicketGroup'].apply(ticket_label)
# sns.barplot(x='TicketGroup',y='Survived',data=all_data,palette='Set3')
# plt.show()

# 缺失值填充
# Age Feature：Age缺失量为263，缺失量较大，用Sex, Title, Pclass
# 三个特征构建随机森林模型，填充年龄缺失值。
age_df = all_data[['Age', 'Pclass', 'Sex', 'Title']]
# print(age_df)
age_df = pd.get_dummies(age_df)
known_age = age_df[age_df.Age.notnull()].as_matrix()
unknown_age = age_df[age_df.Age.isnull()].as_matrix()
# print(unknown_age)
# print(age_df)
y = known_age[:, 0]
x = known_age[:, 1:]
rfr = RandomForestRegressor(random_state=0, n_estimators=100)
rfr.fit(x, y)
predictedAges = rfr.predict(unknown_age[:, 1:])
# print(unknown_age[:,1:])
# print(predictedAges)
all_data.loc[(all_data.Age.isnull()), 'Age'] = predictedAges

# Embarked Feature：Embarked缺失量为2，缺失Embarked信息的乘客的Pclass均为1
# ，且Fare均为80，因为Embarked为C且Pclass为1的乘客的Fare中位数为80，
# 所以缺失值填充为C。
# print(all_data[all_data['Embarked'].isnull()])
# sns.barplot(x='Embarked',y='Fare',hue='Pclass',data=all_data,palette='Set3')
# plt.show()
all_data['Embarked'] = all_data['Embarked'].fillna('C')
# Fare Feature：Fare缺失量为1，缺失Fare信息的乘客的Embarked为S，
# Pclass为3，所以用Embarked为S，Pclass为3的乘客的Fare中位数填充。
fare = all_data[(all_data['Embarked'] == 'S') & (all_data['Pclass'] == 3)].Fare.median()
all_data['Fare'] = all_data['Fare'].fillna(fare)
# 把姓氏相同的乘客划分为同一组，从人数大于一的组中分别提取出每组的妇女儿童和成年男性
all_data['Surname'] = all_data['Name'].apply(lambda x: x.split(',')[0].strip())
Surname_Count = dict(all_data['Surname'].value_counts())
# print(Surname_Count)
all_data['FamilyGroup'] = all_data['Surname'].apply(lambda x: Surname_Count[x])
FemaleChildGroup = all_data.loc[(all_data['FamilyGroup'] >= 2)
                                & ((all_data['Age'] <= 12) | (all_data['Sex'] == 'female'))]
MaleAdultGroup = all_data.loc[(all_data['FamilyGroup'] >= 2) &
                              (all_data['Age'] > 12) & (all_data['Sex'] == 'male')]
# print(FemaleChildGroup)
# 发现绝大部分女性和儿童组的平均存活率都为1或0，即同组的女性和儿童要么全部幸存，要么全部遇难。
# FemaleChild = pd.DataFrame(FemaleChildGroup.groupby('Surname')['Survived'].mean().value_counts())
# FemaleChild.columns = ['GroupCount']
# print(FemaleChild)
# 绝大部分成年男性组的平均存活率也为1或0
# MaleAdult = pd.DataFrame(MaleAdultGroup.groupby('Surname')['Survived'].mean().value_counts())
# MaleAdult.columns = ['GroupCount']
# print(MaleAdult)

# 因为普遍规律是女性和儿童幸存率高，成年男性幸存较低，所以我们把不符合普遍规律的
# 反常组选出来单独处理。把女性和儿童组中幸存率为0的组设置为遇难组，把成年男性组
# 中存活率为1的设置为幸存组，推测处于遇难组的女性和儿童幸存的可能性较低，处于幸
# 存组的成年男性幸存的可能性较高。
FemaleChildGroup = FemaleChildGroup.groupby('Surname')['Survived'].mean()
DeadList = set(FemaleChildGroup[FemaleChildGroup.apply(lambda x: x == 0)].index)
# print(DeadList)
MaleAdultList = MaleAdultGroup.groupby('Surname')['Survived'].mean()
SurvivedList = set(MaleAdultList[MaleAdultList.apply(lambda x: x == 1)].index)
# print(SurvivedList)
# 为了使处于这两种反常组中的样本能够被正确分类，对测试集中处于反常组中的样本的Age，Title，Sex进行惩罚修改
train = all_data.loc[all_data['Survived'].notnull()]
test = all_data.loc[all_data['Survived'].isnull()]
test.loc[(test['Surname'].apply(lambda x: x in DeadList)), 'Sex'] = 'male'
test.loc[(test['Surname'].apply(lambda x: x in DeadList)), 'Age'] = '60'
test.loc[(test['Surname'].apply(lambda x: x in DeadList)), 'Title'] = 'Mr'
test.loc[(test['Surname'].apply(lambda x: x in SurvivedList)), 'Sex'] = 'female'
test.loc[(test['Surname'].apply(lambda x: x in SurvivedList)), 'Age'] = '5'
test.loc[(test['Surname'].apply(lambda x: x in SurvivedList)), 'Title'] = 'Miss'

# 选取特征，转换为数值变量，划分训练集和测试集
all_data = pd.concat([train, test])
all_data = all_data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title',
                     'FamilyLabel', 'Deck', 'TicketGroup']]
all_data = pd.get_dummies(all_data)
train = all_data[all_data['Survived'].notnull()]
test = all_data[all_data['Survived'].isnull()].drop('Survived', axis=1)
x = train.as_matrix()[:, 1:]
y = train.as_matrix()[:, 0]
# .建模和优化
# 1)参数优化
# 用网格搜索自动化选取最优参数，事实上我用网格搜索得到的最优参数是n_estimators = 28，
# max_depth = 6。但是参考另一篇Kernel把参数改为n_estimators = 26，
# max_depth = 6之后交叉验证分数和kaggle评分都有略微提升。
pipe = Pipeline([('select', SelectKBest(k=20)), ('classify', RandomForestClassifier(random_state=10,
                                                                                    max_features='sqrt'))])
param_test = {'classify__n_estimators': list(range(20, 50, 2)),
              'classify__max_depth': list(range(3, 60, 3))}
gsearch = GridSearchCV(estimator=pipe, param_grid=param_test, scoring='roc_auc', cv=10)
gsearch.fit(x, y)
print(gsearch.best_params_, gsearch.best_score_)
#2)训练模型
# select = SelectKBest(k=20)
# clf = RandomForestClassifier(random_state=10, warm_start=True,
#                              n_estimators=26,
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
