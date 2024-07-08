import matplotlib.pyplot as plt  # 导入matplotlib库,用于数据可视化
import numpy as np  # 导入numpy库,用于数据处理和计算
from sklearn.cluster import KMeans  # 导入KMeans算法
from sklearn import datasets  # 导入数据集


iris = datasets.load_iris()  # 加载鸢尾花数据集
x = iris.data[:, :4]  # 选择前四个特征值
print(x.shape)  # 输出(150, 4)，即150个样本，每个样本有4个特征值

# 绘制散点图
plt.scatter(x[:, 0], x[:, 1], c='red', marker='o', label='0')  # 绘制前两个特征值的分布散点图(萼片长度、萼片宽度)
plt.xlabel("sepal length")  # 设置x轴标签
plt.ylabel("sepal width")  # 设置y轴标签
plt.legend()  # 显示图例
plt.show()  # 显示图形
plt.scatter(x[:, 2], x[:, 3], c='blue', marker='x', label='1')  # 绘制后两个特征值的分布散点图(花瓣长度、花瓣宽度)
plt.xlabel("petal length")  # 设置x轴标签
plt.ylabel("petal width")  # 设置y轴标签
plt.legend()  # 显示图例
plt.show()  # 显示图形

# 使用KMeans算法进行聚类
model = KMeans(n_clusters=3, init="k-means++")  # 设置聚类中心数量为3, 算法为k-means++
model.fit(x)  # 训练模型

# 评估指标
print("聚类中心：", model.cluster_centers_)  # 输出聚类中心
print("聚类结果：", model.labels_)  # 输出每个样本的聚类结果
print("SSE={0}".format(model.inertia_))

# 可视化聚类结果
label_prediction = model.labels_  # 获取聚类结果
x0 = x[label_prediction == 0]  # 提取聚类结果为0的样本
x1 = x[label_prediction == 1]  # 提取聚类结果为1的样本
x2 = x[label_prediction == 2]  # 提取聚类结果为2的样本
plt.scatter(x0[:, 0], x0[:, 1], c='red', marker='o', label='0')  # 绘制聚类结果为0的样本的散点图
plt.scatter(x1[:, 0], x1[:, 1], c='blue', marker='x', label='1')  # 绘制聚类结果为1的样本的散点图
plt.scatter(x2[:, 0], x2[:, 1], c='green', marker='+', label='2')  # 绘制聚类结果为2的样本的散点图
plt.xlabel("sepal length")  # 设置x轴标签
plt.ylabel("sepal width")  # 设置y轴标签
plt.legend()  # 显示图例
plt.show()  # 显示图形

plt.scatter(x0[:, 2], x0[:, 3], c='red', marker='o', label='0')  # 绘制聚类结果为0的样本的散点图
plt.scatter(x1[:, 2], x1[:, 3], c='blue', marker='x', label='1')  # 绘制聚类结果为1的样本的散点图
plt.scatter(x2[:, 2], x2[:, 3], c='green', marker='+', label='2')  # 绘制聚类结果为2的样本的散点图
plt.xlabel("petal length")  # 设置x轴标签
plt.ylabel("petal width")  # 设置y轴标签
plt.legend()  # 显示图例
plt.show()  # 显示图形

