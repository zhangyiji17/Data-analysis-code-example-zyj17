import matplotlib.pyplot as plt   # 导入matplotlib库, 用于数据可视化
from sklearn import datasets  # 从sklearn库导入datasets模块, 用于加载数据集
from sklearn.tree import DecisionTreeClassifier, plot_tree  # 从sklearn.tree模块导入DecisionTreeClassifier类, 用于构建决策树模型
import numpy as np   # 导入numpy库, 用于数据处理
from IPython.display import Image   # 导入Image库, 用于显示决策树图形
from sklearn import tree   # 导入tree模块, 用于绘制决策树图形
import pydotplus   # 导入pydotplus库, 用于将决策树图形保存为图像文件
from sklearn.model_selection import train_test_split  # 导入train_test_split函数, 用于数据集划分
from sklearn.pipeline import Pipeline  # 导入Pipeline类, 用于构建数据处理管道
from sklearn.preprocessing import MinMaxScaler  # 导入MinMaxScaler类, 用于数据归一化
from sklearn.feature_selection import SelectKBest  # 导入SelectKBest类, 用于特征选择
from sklearn.decomposition import PCA  # 导入PCA类, 用于特征降维
from sklearn.model_selection import GridSearchCV  # 导入GridSearchCV类, 用于模型超参数调优


# 加载数据集
np.random.seed(0)  # 设置随机种子, 以便结果可重复
iris = datasets.load_iris()  # 加载iris数据集
x = iris.data  # 特征矩阵
y = iris.target  # 目标向量（标签）
indices = np.random.permutation(len(x))  # 随机打乱数据集
iris_x_train = x[indices[:-30]]  # 训练特征矩阵
iris_y_train = y[indices[:-30]]  # 训练目标向量（标签）
iris_x_test = x[indices[-30:]]  # 测试特征矩阵
iris_y_test = y[indices[-30:]]  # 测试目标向量（标签）
# iris_x_train, iris_x_test, iris_y_train, iris_y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 构建决策树模型
clf = DecisionTreeClassifier(max_depth=4)  # 创建决策树分类器对象并设置最大深度为4
clf.fit(iris_x_train, iris_y_train)  # 使用训练数据拟合决策树模型

# 预测测试数据
iris_y_predict = clf.predict(iris_x_test)  # 使用测试数据进行预测

# 可视化决策树
fig, ax = plt.subplots(figsize=(5, 5))  # 创建图形和坐标轴对象
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names,
          filled=True, rounded=True, ax=ax)  # 绘制决策树图形
plt.show()  # 显示图形

# 保存决策树图形为图像文件
# dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names,
#                                 filled=True, rounded=True, special_characters=True)  # 导出决策树图形为DOT格式数据
# graph = pydotplus.graph_from_dot_data(dot_data)  # 将DOT数据转换为图形对象
# Image(graph.create_png())  # 显示图形

# 评估模型性能
score = clf.score(iris_x_test, iris_y_test, sample_weight=None)  # 计算模型准确率
print("模型准确率:", score)
print("预测结果:", iris_y_predict)
print("真实结果:", iris_y_test)

# 深度过拟合比较
max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 最大深度列表
for max_depth in max_depths:
    clf = DecisionTreeClassifier(max_depth=max_depth)  # 创建决策树分类器对象并设置最大深度为当前值
    clf.fit(iris_x_train, iris_y_train)  # 使用训练数据拟合决策树模型
    score = clf.score(iris_x_test, iris_y_test, sample_weight=None)  # 计算模型准确率
    print("最大深度为{}时的模型准确率:".format(max_depth), score)

# 特征重要性
importance = clf.feature_importances_  # 获取特征重要性
for i, v in enumerate(importance):
    print(f'Feature {i}: {v}')  # 打印每个特征的重要性

# 特征筛选
threshold = 0.2  # 设置阈值
selected_features = np.where(importance > threshold)[0]  # 获取重要性大于阈值的特征索引
print("Selected features:", selected_features)

# 参数调优
# pipe = Pipeline([
#     ('nms', MinMaxScaler()),  # 归一化处理
#     ('skb', SelectKBest()),  # 选择特征
#     ('pca', PCA()),  # 主成分分析
#     ('decision', DecisionTreeClassifier(random_state=0))  # 决策树分类器
#     ])

parameters = {
    # "pca__n_components": [0.5, 0.99],  # 设置PCA的n_components参数范围
    # "skb__k": [2, 3],  # 设置SelectKBest的k参数范围
    'max_depth': [1, 2, 3, 4, 5, 6],  # 设置最大深度参数范围
}

gscv = GridSearchCV(clf, param_grid=parameters, cv=5)  # 创建网格搜索交叉验证对象
gscv.fit(iris_x_train, iris_y_train)  # 使用训练数据进行模型训练和参数调优
print("Best parameters found:", gscv.best_params_)  # 打印最佳参数组合
print("Best score found:", gscv.best_score_)  # 打印最佳准确率
print("Best model:", end='')
print(gscv.best_estimator_)

