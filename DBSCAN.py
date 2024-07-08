import numpy as np  # 导入numpy库,用于数据处理和计算
import matplotlib.pyplot as plt  # 导入matplotlib库,用于数据可视化
from sklearn.datasets import make_moons, make_circles, make_blobs  # 导入数据生成器
from sklearn.preprocessing import StandardScaler  # 导入数据标准化器
from sklearn.cluster import DBSCAN  # 导入DBSCAN聚类算法
from sklearn.metrics import silhouette_score  # 导入轮廓系数评估聚类效果
from sklearn.model_selection import GridSearchCV  # 导入网格搜索交叉验证
from sklearn.metrics import make_scorer  # 导入评分器

# https://mp.weixin.qq.com/s/ftyMroFA9mwMu-E6C9x4vg  创建数据集方法

# 生成数据
n_samples = 3500
noisy_moons = make_moons(n_samples=n_samples, noise=0.05)  # 生成具有噪声的月亮形状数据集
noisy_circles = make_circles(n_samples=n_samples, factor=0.5, noise=0.05)  # 生成具有噪声的圆形形状数据集
blobs = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0)  # 生成具有三个簇的随机数据集
random_data = np.random.rand(n_samples, 2), None  # 生成随机数据集

datasets = [noisy_moons, noisy_circles, blobs, random_data]  # 数据集列表

# 标准化数据
datasets = [(StandardScaler().fit_transform(data), labels) for data, labels in datasets]

# 可视化DBSCAN聚类结果
fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # 创建2x2个子图

# 设置DBSCAN参数
eps = 0.3
min_samples = 10

for i, (data, labels) in enumerate(datasets):
    ax = axes[i // 2, i % 2]  # 获取当前子图对象
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # 创建DBSCAN模型
    clusters = dbscan.fit_predict(data)  # 训练模型并获取聚类结果

    # 评估指标
    if len(set(clusters)) > 1:  # 至少要有两个簇
        silhouette_avg = silhouette_score(data, clusters)  # 轮廓系数
        print(f"Dataset {i + 1} - Silhouette Score: {silhouette_avg:.3f}")

    # 可视化聚类结果
    unique_labels = set(clusters)  # 获取所有簇标签
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]  # 生成颜色列表

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # 设置噪声点的颜色为黑色
        class_member_mask = (clusters == k)  # 获取当前簇的成员
        xy = data[class_member_mask]  # 获取当前簇的数据点
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)  # 绘制数据点

    ax.set_title(f'Dataset {i + 1}')
    ax.set_title(f'DBSCAN (eps={eps:.2f}, min_samples={min_samples})')  # 设置子图标题
    ax.set_xlim([-2, 2])  # 设置x轴范围
    ax.set_ylim([-2, 2])  # 设置y轴范围

plt.tight_layout()  # 调整子图布局
plt.show()  # 显示图形


# 网格搜索DBSCAN参数
def dbscan_silhouette_scorer(estimator, x):
    labels = estimator.fit_predict(x)  # 拟合数据并获取簇标签
    if len(set(labels)) > 1:  # 至少要有两个簇
        return silhouette_score(x, labels)  # 计算轮廓系数
    else:
        return 0.0  # 只有一个簇，轮廓系数为0


# 网格搜索参数范围
param_grid = {
    'eps': np.arange(0.1, 0.5, 0.1),
    'min_samples': np.arange(5, 20, 5)
    }

best_params = []  # 存储最佳参数

# 遍历每个数据集
for data, label in datasets:
    grid_search = GridSearchCV(DBSCAN(), param_grid, scoring=make_scorer(dbscan_silhouette_scorer))  # 创建网格搜索交叉验证对象
    grid_search.fit(data)  # 拟合数据
    best_params.append(grid_search.best_params_)  # 获取最佳参数
    print(f"Best parameters for each dataset:{grid_search.best_params_}")

# 使用最佳参数进行DBSCAN聚类并可视化
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for i, (data, labels) in enumerate(datasets):
    ax = axes[i // 2, i % 2]  # 获取子图
    best_eps = best_params[i]['eps']  # 获取最佳eps值
    best_min_samples = best_params[i]['min_samples']  # 获取最佳min_samples值

    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)  # 使用最佳参数创建DBSCAN模型
    clusters = dbscan.fit_predict(data)  # 拟合数据并获取簇标签

    # 计算轮廓系数
    if len(set(clusters)) > 1:  # 至少要有两个簇
        silhouette_avg = silhouette_score(data, clusters)  # 计算轮廓系数
        print(f"Dataset {i + 1} - Silhouette Score: {silhouette_avg:.3f}")

    # 可视化聚类结果
    unique_labels = set(clusters)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # 设置噪声点的颜色为黑色

        class_member_mask = (clusters == k)  # 获取当前簇的样本
        xy = data[class_member_mask]  # 获取当前簇的样本坐标
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)  # 绘制簇

    ax.set_title(f'DBSCAN (eps={best_eps:.2f}, min_samples={best_min_samples})')  # 设置子图标题
    ax.set_xlim([-2, 2])  # 设置x轴范围
    ax.set_ylim([-2, 2])  # 设置y轴范围

plt.tight_layout()  # 调整子图之间的间距
plt.show()  # 显示图形
