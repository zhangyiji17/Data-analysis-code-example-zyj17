import numpy as np  # 导入numpy库,用于科学计算
from sklearn import datasets  # 导入数据集
from sklearn.model_selection import train_test_split  # 导入数据集划分函数
from sklearn.naive_bayes import MultinomialNB, GaussianNB  # 导入朴素贝叶斯分类器
from sklearn.metrics import accuracy_score  # 导入准确率评估函数
from matplotlib import pyplot as plt  # 导入matplotlib库,用于数据可视化


# 加载数据集
iris = datasets.load_iris()
x = iris.data  # 特征值
y = iris.target  # 目标值
# iris_x = x[:, :2]  # 只取前两个特征值
# iris_y = y

# 数据集划分，训练集：测试集=7:3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)  # 划分训练集和测试集

# 创建朴素贝叶斯分类器
clf = GaussianNB()

# 训练模型
clf.fit(x_train, y_train)

# 预测测试集
y_predict = clf.predict(x_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_predict)
print("准确率:", accuracy)

# 可视化数据集
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Iris dataset')
plt.show()


