import jerry_numpy as np
import random


# py实现的logistic regression算法 彭亮教程

def gradientDesent(x, y, theta, alpha, m, num_iterations):
    x_trans = x.transpose()
    for i in range(num_iterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | cost: %f" % (i, cost))
        gradient = np.dot(x_trans, loss) / m
        theta = theta - alpha * gradient
    return theta


def genData(numPoints, bias, variacce):
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    for i in range(numPoints):
        x[i][0] = 1
        x[i][1] = i
        y[i] = (i + bias) + random.uniform(0, 1) * variacce
    return x, y


x, y = genData(100, 25, 10)
theta = np.ones(shape=2)
alpha = 0.0005
num_iterations = 1000
print(gradientDesent(x, y, theta, alpha, len(y), num_iterations))
