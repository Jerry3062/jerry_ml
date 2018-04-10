# 检视不同svm的gamma参数
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_digits  # digits数据集
from sklearn.svm import SVC  # Support Vector Classifier
import matplotlib.pyplot as plt  # 可视化模块
import jerry_numpy as np

digits = load_digits()
X = digits.data
y = digits.target

param_range = np.logspace(-6, -2, 10)

print(param_range)
print(type(param_range))

train_loss, test_loss = validation_curve(
    SVC(), X, y, param_name='gamma', param_range=param_range,
    cv=10, scoring='neg_mean_squared_error')

# 平均每一轮所得到的平均方差(共5轮，分别为样本10%、25%、50%、75%、100%)
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(param_range, train_loss_mean, 'o-', color="r",
         label="Training")
plt.plot(param_range, test_loss_mean, 'o-', color="g",
         label="Cross-validation")

plt.xlabel("gamma")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()
