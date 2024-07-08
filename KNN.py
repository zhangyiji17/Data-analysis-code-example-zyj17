import numpy as np  # 导入numpy库, 用于科学计算
import matplotlib.pyplot as plt   # 导入matplotlib库, 用于数据可视化
from sklearn import datasets  # 从sklearn库导入datasets模块, 用于加载数据集
from sklearn.model_selection import train_test_split  # 导入train_test_split函数, 用于数据集划分
from sklearn.preprocessing import StandardScaler  # 导入StandardScaler类, 用于数据标准化
from sklearn.neighbors import KNeighborsClassifier  # 导入KNeighborsClassifier类, 用于构建KNN模型
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # 导入评估指标函数, 用于评估模型性能
from sklearn.model_selection import cross_val_score  # 导入交叉验证函数, 用于模型性能评估

# 加载数据集
iris = datasets.load_iris()
x = iris.data  # 特征矩阵
y = iris.target  # 目标向量（标签）

# 数据集划分，训练集：测试集=8:2
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()  # 创建StandardScaler对象
x_train = scaler.fit_transform(x_train)  # 拟合训练集数据并转换
x_test = scaler.transform(x_test)  # 使用训练集得到的均值和标准差直接转换测试集数据

# 构建KNN模型
knn = KNeighborsClassifier(n_neighbors=3)  # 创建KNeighborsClassifier对象，指定邻居数量(K值=3)
knn.fit(x_train, y_train)  # 训练模型

# 预测测试集
predict = knn.predict(x_test)

# 评估模型性能
print("Accuracy:", accuracy_score(y_test, predict))  # 准确率
print(classification_report(y_test, predict))  # 分类报告
print(confusion_matrix(y_test, predict))  # 混淆矩阵

# 交叉验证
k_range = range(1, 21)  # 指定K值范围
k_scores = []  # 存储不同K值下的准确率

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)  # 创建KNeighborsClassifier对象，指定邻居数量(K值)
    scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')  # 交叉验证，指定10折交叉验证
    k_scores.append(scores.mean())  # 存储平均准确率

# 绘制K值与准确率关系图
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')  # 设置x轴标签
plt.ylabel('Testing Accuracy')  # 设置y轴标签
plt.title('Accuracy vs K Value')  # 设置图表标题
plt.show()  # 显示图表

