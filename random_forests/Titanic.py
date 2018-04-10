# coding: utf-8

# In[2]:


# 这个ipython notebook主要是我解决Kaggle Titanic问题的思路和过程

import pandas as pd  # 数据分析
import numpy as np  # 科学计算
from pandas import Series, DataFrame

data_train = pd.read_csv("Train.csv")
data_train.columns
# data_train[data_train.Cabin.notnull()]['Survived'].value_counts()


# In[3]:


data_train.info()
# 我们发现有一些列，比如说Cabin，有非常多的缺失值
# 另外一些我们感觉在此场景中会有影响的属性，比如Age，也有一些缺失值


# In[4]:


data_train.describe()

# In[5]:


import matplotlib.pyplot as plt

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

plt.subplot2grid((2, 3), (0, 0))  # 在一张大图里分列几个小图
data_train.Survived.value_counts().plot(kind='bar')  # plots a bar graph of those who surived vs those who did not.
plt.title(u"获救情况 (1为获救)")  # puts a title on our graph
plt.ylabel(u"人数")

plt.subplot2grid((2, 3), (0, 1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel(u"人数")
plt.title(u"乘客等级分布")

plt.subplot2grid((2, 3), (0, 2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"年龄")  # sets the y axis lable
plt.grid(b=True, which='major', axis='y')  # formats the grid line style of our graphs
plt.title(u"按年龄看获救分布 (1为获救)")

plt.subplot2grid((2, 3), (1, 0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(
    kind='kde')  # plots a kernel desnsity estimate of the subset of the 1st class passanges's age
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"年龄")  # plots an axis lable
plt.ylabel(u"密度")
plt.title(u"各等级的乘客年龄分布")
plt.legend((u'头等舱', u'2等舱', u'3等舱'), loc='best')  # sets our legend for our graph.

plt.subplot2grid((2, 3), (1, 2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数")
plt.show()

# In[6]:


# 看看各乘客等级的获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各乘客等级的获救情况")
plt.xlabel(u"乘客等级")
plt.ylabel(u"人数")

plt.show()

# In[7]:


# 看看各登录港口的获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各登录港口乘客的获救情况")
plt.xlabel(u"登录港口")
plt.ylabel(u"人数")

plt.show()

# In[8]:


# 看看各性别的获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df = pd.DataFrame({u'男性': Survived_m, u'女性': Survived_f})
df.plot(kind='bar', stacked=True)
plt.title(u"按性别看获救情况")
plt.xlabel(u"性别")
plt.ylabel(u"人数")
plt.show()

# In[9]:


# 然后我们再来看看各种舱级别情况下各性别的获救情况
fig = plt.figure()
fig.set(alpha=0.65)  # 设置图像透明度，无所谓
plt.title(u"根据舱等级和性别的获救情况")

ax1 = fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar',
                                                                                            label="female highclass",
                                                                                            color='#FA2479')
ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)
ax1.legend([u"女性/高级舱"], loc='best')

ax2 = fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar',
                                                                                            label='female, low class',
                                                                                            color='pink')
ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"女性/低级舱"], loc='best')

ax3 = fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar',
                                                                                          label='male, high class',
                                                                                          color='lightblue')
ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/高级舱"], loc='best')

ax4 = fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar',
                                                                                          label='male low class',
                                                                                          color='steelblue')
ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/低级舱"], loc='best')

plt.show()

# In[10]:


g = data_train.groupby(['SibSp', 'Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
df

# In[11]:


g = data_train.groupby(['Parch', 'Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
df

# In[12]:


# ticket是船票编号，应该是unique的，和最后的结果没有太大的关系，不纳入考虑的特征范畴
# cabin只有204个乘客有值，我们先看看它的一个分布
data_train.Cabin.value_counts()

# In[13]:


# cabin的值计数太分散了，绝大多数Cabin值只出现一次。感觉上作为类目，加入特征未必会有效
# 那我们一起看看这个值的有无，对于survival的分布状况，影响如何吧
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df = pd.DataFrame({u'有': Survived_cabin, u'无': Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.title(u"按Cabin有无看获救情况")
plt.xlabel(u"Cabin有无")
plt.ylabel(u"人数")
plt.show()

# 似乎有cabin记录的乘客survival比例稍高，那先试试把这个值分为两类，有cabin值/无cabin值，一会儿加到类别特征好了


# In[14]:


# 我们发现Age这个属性也有177个乘客没有记录。
# 通常遇到缺值的情况，我们会有几种常见的处理方式
# 1. 如果缺值的样本占总数比例极高，我们可能就直接舍弃了，作为特征加入的话，可能反倒带入noise，影响最后的结果了
# 2. 如果缺值的样本适中，而该属性非连续值特征属性，那就把NaN作为一个类别，加到类别特征中
# 3. 如果缺值的样本适中，而该属性为连续值特征属性，有时候我们会考虑给定一个step(比如这里的age，我们可以考虑每隔2/3岁为一个步长)，然后把它离散化，之后把NaN作为一个type加到属性类目中
# 4. 有些情况下，缺失的值个数并不是特别多，那我们也可以试着根据已有的值，拟合一下数据，补充上
# 本例中，3和4应该都是可行的，我们试试4
# 我们这里用scikit-learn中的RandomForest来拟合一下缺失的年龄数据
# 注：RandomForest是一个用在原始数据中做不同采样，建立多颗DecisionTree，再进行vote取结果的机器学习算法，我们之后会介绍到
from sklearn.ensemble import RandomForestRegressor


### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
data_train

# In[565]:


# 因为逻辑回归建模时，需要输入的特征都是数值型特征
# 我们先对类目型的特征离散/因子化
# 以Cabin为例，原本一个属性维度，因为其取值可以是['yes','no']，而将其平展开为'Cabin_yes','Cabin_no'两个属性
# 原本Cabin取值为yes的，在此处的'Cabin_yes'下取值为1，在'Cabin_no'下取值为0
# 原本Cabin取值为no的，在此处的'Cabin_yes'下取值为0，在'Cabin_no'下取值为1
# 我们使用pandas的get_dummies来完成这个工作，并拼接在原来的data_train之上，如下所示
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df

# In[15]:


# 接下来我们要接着做一些数据预处理的工作，比如scaling，将一些变化幅度较大的特征化到[-1,1]之内
# 这样可以加速logistic regression的收敛
import sklearn.preprocessing as preprocessing

scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'])
df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
fare_scale_param = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)
df

# In[573]:


# 我们把需要的feature字段取出来，转成numpy格式，使用scikit-learn中的LogisticRegression建模
from sklearn import linear_model

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)

clf

# In[574]:


X.shape

# In[579]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve


# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


plot_learning_curve(clf, u"学习曲线", X, y)

# In[569]:


data_test = pd.read_csv("/Users/MLS/Downloads/test.csv")
data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'], age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'], fare_scale_param)
df_test

# In[371]:


test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
result.to_csv("/Users/MLS/Downloads/logistic_regression_predictions.csv", index=False)

# In[515]:


pd.read_csv("/Users/MLS/Downloads/logistic_regression_predictions.csv")

# In[359]:


# 0.76555，恩，结果还不错。毕竟，这只是我们简单分析过后出的一个baseline系统嘛
# Andrew Ng老师说的第一步到此我们就做完了，你以为故事到这里就结束了吗？图样图森破啊，这只是第一步好咩！！接下来，我们就该看看如何优化baseline系统了
# 看过Andrew Ng老师的machine Learning 课程的同学们，知道，我们应该分析分析模型现在的状态了，是 过/欠拟合？，对，就是下面这个很神奇的图，以便了解下一步做什么样的操作可能是有效的
# 不过在现在的场景下，先不着急做这个事情，我们这个baseline系统有些粗糙，先再挖掘挖掘
# 1. 比如说Name和Ticket两个属性被我们完整舍弃了(好吧，其实是一开始我们对于这种，每一条记录都是一个完全不同的值的属性，并没有很直接的处理方式)
# 2. 比如说，我们想想，年龄的拟合本身也未必是一件非常靠谱的事情
# 3. 另外，以我们的日常经验，小盆友和老人可能得到的照顾会多一些，这样看的话，年龄作为一个连续值，给一个固定的系数，似乎体现不出两头受照顾的实际情况，所以，说不定我们把年龄离散化，按区段分作类别属性会更合适一些
# 那怎么样才知道，哪些地方可以优化，哪些优化的方法是promising的呢？
# 是的，要做交叉验证(cross validation)!要做交叉验证(cross validation)!要做交叉验证(cross validation)!重要的事情说3编！！！
# 因为test.csv里面并没有Survived这个字段(好吧，这是废话，这明明就是我们要预测的结果)，我们无法在这份数据上评定我们算法在该场景下的效果。。。
# 别，哥们儿，你没那个想法吧『每做一次调整就make a submission，然后根据结果来判定这次调整的好坏』，这。。。其实是行不通的。。。
# 我们通常情况下，这么做cross validation：把train.csv分成两部分，一部分用于训练我们需要的模型，另外一部分数据上看我们预测算法的效果。
# 我们用scikit-learn的cross_validation来完成这个工作


# In[493]:


pd.DataFrame({"columns": list(train_df.columns)[1:], "coef": list(clf.coef_.T)})

# In[ ]:


# 上面的系数和最后的结果是一个正相关的关系


# In[560]:


from sklearn import cross_validation

# 简单看看打分情况
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
X = all_data.as_matrix()[:, 1:]
y = all_data.as_matrix()[:, 0]
print(cross_validation.cross_val_score(clf, X, y, cv=5))

# 分割数据
split_train, split_cv = cross_validation.train_test_split(
    df, test_size=0.3, random_state=0)
train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# 生成模型
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(train_df.as_matrix()[:, 1:], train_df.as_matrix()[:, 0])

# 对cross validation数据进行预测

cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(cv_df.as_matrix()[:, 1:])
split_cv[predictions != cv_df.as_matrix()[:, 0]].drop()

# In[562]:


# 去除预测错误的case看原始dataframe数据
# split_cv['PredictResult'] = predictions
origin_data_train = pd.read_csv("/Users/MLS/Downloads/Train.csv")
bad_cases = origin_data_train.loc[
    origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:, 0]]['PassengerId'].values)]
bad_cases

# In[440]:


data_train[data_train['Name'].str.contains("Major")]

# In[431]:


# In[546]:


data_train = pd.read_csv("/Users/MLS/Downloads/Train.csv")
data_train['Sex_Pclass'] = data_train.Sex + "_" + data_train.Pclass.map(str)

from sklearn.ensemble import RandomForestRegressor


### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
dummies_Sex_Pclass = pd.get_dummies(data_train['Sex_Pclass'], prefix='Sex_Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass, dummies_Sex_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Sex_Pclass'], axis=1, inplace=True)
import sklearn.preprocessing as preprocessing

scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'])
df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
fare_scale_param = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)

from sklearn import linear_model

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)
clf

# In[543]:


data_test = pd.read_csv("/Users/MLS/Downloads/test.csv")
data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
data_test['Sex_Pclass'] = data_test.Sex + "_" + data_test.Pclass.map(str)
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')
dummies_Sex_Pclass = pd.get_dummies(data_test['Sex_Pclass'], prefix='Sex_Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass, dummies_Sex_Pclass],
                    axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Sex_Pclass'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'], age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'], fare_scale_param)
df_test

# In[545]:


test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
result.to_csv("/Users/MLS/Downloads/logistic_regression_predictions2.csv", index=False)

# In[550]:


# In[ ]:


from sklearn.ensemble import BaggingRegressor

train_df = df.filter(
    regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到BaggingRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=10, max_samples=0.8, max_features=1.0, bootstrap=True,
                               bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
predictions = bagging_clf.predict(test)
result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
result.to_csv("/Users/MLS/Downloads/logistic_regression_predictions2.csv", index=False)

# In[581]:


# Titatic competitor usign pandas and scikit library
import numpy as np
import pandas as pd
from pandas import DataFrame
from patsy import dmatrices
import string
from operator import itemgetter
# json library for settings file
import json
# import the machine learning library that holds the randomforest
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import classification_report
# joblib library for serialization
from sklearn.externals import joblib

##Read configuration parameters

train_file = "/Users/Hanxiaoyang/Downloads/train.csv"
MODEL_PATH = "/Users/Hanxiaoyang/Downloads/"
test_file = "/Users/Hanxiaoyang/Downloads/test.csv"
SUBMISSION_PATH = "/Users/Hanxiaoyang/Downloads/"
seed = 0

print
train_file, seed


# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            score.mean_validation_score,
            np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


###utility to clean and munge data
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if string.find(big_string, substring) != -1:
            return substring
    print
    big_string
    return np.nan


le = preprocessing.LabelEncoder()
enc = preprocessing.OneHotEncoder()


def clean_and_munge_data(df):
    # setting silly values to nan
    df.Fare = df.Fare.map(lambda x: np.nan if x == 0 else x)
    # creating a title column from name
    title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                  'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme', 'Countess',
                  'Don', 'Jonkheer']
    df['Title'] = df['Name'].map(lambda x: substrings_in_string(x, title_list))

    # replacing all titles with mr, mrs, miss, master
    def replace_titles(x):
        title = x['Title']
        if title in ['Mr', 'Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Master']:
            return 'Master'
        elif title in ['Countess', 'Mme', 'Mrs']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms', 'Miss']:
            return 'Miss'
        elif title == 'Dr':
            if x['Sex'] == 'Male':
                return 'Mr'
            else:
                return 'Mrs'
        elif title == '':
            if x['Sex'] == 'Male':
                return 'Master'
            else:
                return 'Miss'
        else:
            return title

    df['Title'] = df.apply(replace_titles, axis=1)

    # Creating new family_size column
    df['Family_Size'] = df['SibSp'] + df['Parch']
    df['Family'] = df['SibSp'] * df['Parch']

    # imputing nan values
    df.loc[(df.Fare.isnull()) & (df.Pclass == 1), 'Fare'] = np.median(df[df['Pclass'] == 1]['Fare'].dropna())
    df.loc[(df.Fare.isnull()) & (df.Pclass == 2), 'Fare'] = np.median(df[df['Pclass'] == 2]['Fare'].dropna())
    df.loc[(df.Fare.isnull()) & (df.Pclass == 3), 'Fare'] = np.median(df[df['Pclass'] == 3]['Fare'].dropna())

    df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    df['AgeFill'] = df['Age']
    mean_ages = np.zeros(4)
    mean_ages[0] = np.average(df[df['Title'] == 'Miss']['Age'].dropna())
    mean_ages[1] = np.average(df[df['Title'] == 'Mrs']['Age'].dropna())
    mean_ages[2] = np.average(df[df['Title'] == 'Mr']['Age'].dropna())
    mean_ages[3] = np.average(df[df['Title'] == 'Master']['Age'].dropna())
    df.loc[(df.Age.isnull()) & (df.Title == 'Miss'), 'AgeFill'] = mean_ages[0]
    df.loc[(df.Age.isnull()) & (df.Title == 'Mrs'), 'AgeFill'] = mean_ages[1]
    df.loc[(df.Age.isnull()) & (df.Title == 'Mr'), 'AgeFill'] = mean_ages[2]
    df.loc[(df.Age.isnull()) & (df.Title == 'Master'), 'AgeFill'] = mean_ages[3]

    df['AgeCat'] = df['AgeFill']
    df.loc[(df.AgeFill <= 10), 'AgeCat'] = 'child'
    df.loc[(df.AgeFill > 60), 'AgeCat'] = 'aged'
    df.loc[(df.AgeFill > 10) & (df.AgeFill <= 30), 'AgeCat'] = 'adult'
    df.loc[(df.AgeFill > 30) & (df.AgeFill <= 60), 'AgeCat'] = 'senior'

    df.Embarked = df.Embarked.fillna('S')

    # Special case for cabins as nan may be signal
    df.loc[df.Cabin.isnull() == True, 'Cabin'] = 0.5
    df.loc[df.Cabin.isnull() == False, 'Cabin'] = 1.5
    # Fare per person

    df['Fare_Per_Person'] = df['Fare'] / (df['Family_Size'] + 1)

    # Age times class

    df['AgeClass'] = df['AgeFill'] * df['Pclass']
    df['ClassFare'] = df['Pclass'] * df['Fare_Per_Person']

    df['HighLow'] = df['Pclass']
    df.loc[(df.Fare_Per_Person < 8), 'HighLow'] = 'Low'
    df.loc[(df.Fare_Per_Person >= 8), 'HighLow'] = 'High'

    le.fit(df['Sex'])
    x_sex = le.transform(df['Sex'])
    df['Sex'] = x_sex.astype(np.float)

    le.fit(df['Ticket'])
    x_Ticket = le.transform(df['Ticket'])
    df['Ticket'] = x_Ticket.astype(np.float)

    le.fit(df['Title'])
    x_title = le.transform(df['Title'])
    df['Title'] = x_title.astype(np.float)

    le.fit(df['HighLow'])
    x_hl = le.transform(df['HighLow'])
    df['HighLow'] = x_hl.astype(np.float)

    le.fit(df['AgeCat'])
    x_age = le.transform(df['AgeCat'])
    df['AgeCat'] = x_age.astype(np.float)

    le.fit(df['Embarked'])
    x_emb = le.transform(df['Embarked'])
    df['Embarked'] = x_emb.astype(np.float)

    df = df.drop(['PassengerId', 'Name', 'Age', 'Cabin'], axis=1)  # remove Name,Age and PassengerId

    return df


# read data
traindf = pd.read_csv(train_file)
##clean data
df = clean_and_munge_data(traindf)
########################################formula################################

formula_ml = 'Survived~Pclass+C(Title)+Sex+C(AgeCat)+Fare_Per_Person+Fare+Family_Size'

y_train, x_train = dmatrices(formula_ml, data=df, return_type='dataframe')
y_train = np.asarray(y_train).ravel()
print
y_train.shape, x_train.shape

##select a train and test set
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=seed)
# instantiate and fit our model

clf = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=5, min_samples_split=1,
                             min_samples_leaf=1, max_features='auto', bootstrap=False, oob_score=False, n_jobs=1,
                             random_state=seed,
                             verbose=0)

###compute grid search to find best paramters for pipeline
param_grid = dict()
##classify pipeline
pipeline = Pipeline([('clf', clf)])
grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=3, scoring='accuracy',
                           cv=StratifiedShuffleSplit(Y_train, n_iter=10, test_size=0.2, train_size=None, indices=None,
                                                     random_state=seed, n_iterations=None)).fit(X_train, Y_train)
# Score the results
###print result
print("Best score: %0.3f" % grid_search.best_score_)
print(grid_search.best_estimator_)
report(grid_search.grid_scores_)

print('-----grid search end------------')
print('on all train set')
scores = cross_val_score(grid_search.best_estimator_, x_train, y_train, cv=3, scoring='accuracy')
print
scores.mean(), scores
print('on test set')
scores = cross_val_score(grid_search.best_estimator_, X_test, Y_test, cv=3, scoring='accuracy')
print
scores.mean(), scores

# Score the results

print(classification_report(Y_train, grid_search.best_estimator_.predict(X_train)))
print('test data')
print(classification_report(Y_test, grid_search.best_estimator_.predict(X_test)))

# serialize training
model_file = MODEL_PATH + 'model-rf.pkl'
joblib.dump(grid_search.best_estimator_, model_file)

# In[583]:


# Titatic competitor usign pandas and scikit library
import numpy as np
import pandas as pd
from pandas import DataFrame
import string
from operator import itemgetter
# json library for settings file
import json
# import the machine learning library that holds the randomforest
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn import preprocessing
# joblib library for serialization
from sklearn.externals import joblib

# Read data and configuration parameters#

train_file = "/Users/Hanxiaoyang/Downloads/train.csv"
MODEL_PATH = "/Users/Hanxiaoyang/Downloads/"
test_file = "/Users/Hanxiaoyang/Downloads/test.csv"
SUBMISSION_PATH = "/Users/Hanxiaoyang/Downloads/"
seed = 0

print
test_file, seed


# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            score.mean_validation_score,
            np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


###utility to clean and munge data
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if string.find(big_string, substring) != -1:
            return substring
    print
    big_string
    return np.nan


le = preprocessing.LabelEncoder()
enc = preprocessing.OneHotEncoder()


def clean_and_munge_data(df):
    # setting silly values to nan
    df.Fare = df.Fare.map(lambda x: np.nan if x == 0 else x)
    # creating a title column from name
    title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                  'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme', 'Countess',
                  'Don', 'Jonkheer']
    df['Title'] = df['Name'].map(lambda x: substrings_in_string(x, title_list))

    # replacing all titles with mr, mrs, miss, master
    def replace_titles(x):
        title = x['Title']
        if title in ['Mr', 'Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Master']:
            return 'Master'
        elif title in ['Countess', 'Mme', 'Mrs']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms', 'Miss']:
            return 'Miss'
        elif title == 'Dr':
            if x['Sex'] == 'Male':
                return 'Mr'
            else:
                return 'Mrs'
        elif title == '':
            if x['Sex'] == 'Male':
                return 'Master'
            else:
                return 'Miss'
        else:
            return title

    df['Title'] = df.apply(replace_titles, axis=1)

    # Creating new family_size column
    df['Family_Size'] = df['SibSp'] + df['Parch']
    df['Family'] = df['SibSp'] * df['Parch']

    # imputing nan values
    df.loc[(df.Fare.isnull()) & (df.Pclass == 1), 'Fare'] = np.median(df[df['Pclass'] == 1]['Fare'].dropna())
    df.loc[(df.Fare.isnull()) & (df.Pclass == 2), 'Fare'] = np.median(df[df['Pclass'] == 2]['Fare'].dropna())
    df.loc[(df.Fare.isnull()) & (df.Pclass == 3), 'Fare'] = np.median(df[df['Pclass'] == 3]['Fare'].dropna())

    df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    df['AgeFill'] = df['Age']
    mean_ages = np.zeros(4)
    mean_ages[0] = np.average(df[df['Title'] == 'Miss']['Age'].dropna())
    mean_ages[1] = np.average(df[df['Title'] == 'Mrs']['Age'].dropna())
    mean_ages[2] = np.average(df[df['Title'] == 'Mr']['Age'].dropna())
    mean_ages[3] = np.average(df[df['Title'] == 'Master']['Age'].dropna())
    df.loc[(df.Age.isnull()) & (df.Title == 'Miss'), 'AgeFill'] = mean_ages[0]
    df.loc[(df.Age.isnull()) & (df.Title == 'Mrs'), 'AgeFill'] = mean_ages[1]
    df.loc[(df.Age.isnull()) & (df.Title == 'Mr'), 'AgeFill'] = mean_ages[2]
    df.loc[(df.Age.isnull()) & (df.Title == 'Master'), 'AgeFill'] = mean_ages[3]

    df['AgeCat'] = df['AgeFill']
    df.loc[(df.AgeFill <= 10), 'AgeCat'] = 'child'
    df.loc[(df.AgeFill > 60), 'AgeCat'] = 'aged'
    df.loc[(df.AgeFill > 10) & (df.AgeFill <= 30), 'AgeCat'] = 'adult'
    df.loc[(df.AgeFill > 30) & (df.AgeFill <= 60), 'AgeCat'] = 'senior'

    df.Embarked = df.Embarked.fillna('S')

    # Special case for cabins as nan may be signal
    df.loc[df.Cabin.isnull() == True, 'Cabin'] = 0.5
    df.loc[df.Cabin.isnull() == False, 'Cabin'] = 1.5
    # Fare per person

    df['Fare_Per_Person'] = df['Fare'] / (df['Family_Size'] + 1)

    # Age times class

    df['AgeClass'] = df['AgeFill'] * df['Pclass']
    df['ClassFare'] = df['Pclass'] * df['Fare_Per_Person']

    df['HighLow'] = df['Pclass']
    df.loc[(df.Fare_Per_Person < 8), 'HighLow'] = 'Low'
    df.loc[(df.Fare_Per_Person >= 8), 'HighLow'] = 'High'

    le.fit(df['Sex'])
    x_sex = le.transform(df['Sex'])
    df['Sex'] = x_sex.astype(np.float)

    le.fit(df['Ticket'])
    x_Ticket = le.transform(df['Ticket'])
    df['Ticket'] = x_Ticket.astype(np.float)

    le.fit(df['Title'])
    x_title = le.transform(df['Title'])
    df['Title'] = x_title.astype(np.float)

    le.fit(df['HighLow'])
    x_hl = le.transform(df['HighLow'])
    df['HighLow'] = x_hl.astype(np.float)

    le.fit(df['AgeCat'])
    x_age = le.transform(df['AgeCat'])
    df['AgeCat'] = x_age.astype(np.float)

    le.fit(df['Embarked'])
    x_emb = le.transform(df['Embarked'])
    df['Embarked'] = x_emb.astype(np.float)

    df = df.drop(['PassengerId', 'Name', 'Age', 'Cabin'], axis=1)  # remove Name,Age and PassengerId

    return df


# read data

testdf = pd.read_csv(test_file)

ID = testdf['PassengerId']
##clean data
df_test = clean_and_munge_data(testdf)
df_test['Survived'] = [0 for x in range(len(df_test))]

print
df_test.shape
########################################formula################################

formula_ml = 'Survived~Pclass+C(Title)+Sex+C(AgeCat)+Fare_Per_Person+Fare+Family_Size'

y_p, x_test = dmatrices(formula_ml, data=df_test, return_type='dataframe')
y_p = np.asarray(y_p).ravel()
print
y_p.shape, x_test.shape
# serialize training
model_file = MODEL_PATH + 'model-rf.pkl'
clf = joblib.load(model_file)
####estimate prediction on test data set
y_p = clf.predict(x_test).astype(int)
print
y_p.shape

outfile = SUBMISSION_PATH + 'prediction-BS.csv'
dfjo = DataFrame(dict(Survived=y_p, PassengerId=ID), columns=['PassengerId', 'Survived'])
dfjo.to_csv(outfile, index_label=None, index_col=False, index=False)
