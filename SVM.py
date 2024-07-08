import numpy as np  # 导入NumPy库，用于数据处理
import matplotlib.pyplot as plt  # 导入matplotlib库，用于数据可视化
from mpl_toolkits.mplot3d import Axes3D  # 导入mpl_toolkits库，用于3D可视化
from sklearn import datasets  # 从sklearn库导入datasets模块, 用于加载数据集
from sklearn.model_selection import train_test_split  # 从sklearn库导入train_test_split模块, 用于数据集划分
from sklearn.preprocessing import StandardScaler  # 从sklearn库导入StandardScaler模块, 用于数据标准化
from sklearn.svm import SVC  # 从sklearn库导入SVC模块, 用于支持向量机分类器
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # 从sklearn库导入评估指标模块, 用于模型评估

# 加载数据集
iris = datasets.load_iris()  # 加载iris数据集
x = iris.data  # 获取特征数据
y = iris.target  # 获取目标数据

# 为数据添加噪声
random_state = np.random.RandomState(0)  # 创建随机数生成器
n_samples, n_features = x.shape  # 获取特征数据的形状
x = np.c_[x, random_state.randn(n_samples, 2 * n_features)]  # 添加噪声

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)  # 划分训练集和测试集

# 数据标准化
scaler = StandardScaler()  # 创建标准化器
x_train = scaler.fit_transform(x_train)  # 对训练集进行标准化
x_test = scaler.transform(x_test)  # 对测试集进行标准化

# 创建支持向量机分类器
clf = SVC(kernel='rbf', C=1.0, gamma='scale')  # 创建支持向量机分类器, 参数C表示正则化强度, 参数gamma表示核函数的带宽

# 训练模型
clf.fit(x_train, y_train)  # 训练模型

# 预测测试集
y_predict = clf.predict(x_test)  # 预测测试集

# 模型评估
print("Accuracy:", accuracy_score(y_test, y_predict))  # 打印准确率
print("Classification Report:\n", classification_report(y_test, y_predict))  # 打印分类报告
print("Confusion Matrix:\n", confusion_matrix(y_test, y_predict))  # 打印混淆矩阵


# 可视化数据
def plot_decision_boundary(x, y, classifier):
    h = .02  # 设置网格大小
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1  # 设置x轴范围
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1  # 设置y轴范围
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))  # 创建网格点
    z = classifier.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())])  # 预测网格点
    z = z.reshape(xx.shape)  # 重塑预测结果
    fig = plt.figure(figsize=(10, 8))  # 创建图形对象
    ax = fig.add_subplot(111, projection='3d')  # 创建3D子图
    ax.contourf(xx, yy, z, alpha=0.5, cmap='coolwarm')  # 绘制等高线图
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, cmap='viridis')  # 绘制数据点
    ax.set_xlabel('Sepal Length')  # 设置x轴标签
    ax.set_ylabel('Sepal Width')  # 设置y轴标签
    ax.set_zlabel('Petal Length')  # 设置z轴标签
    plt.show()  # 显示图形


# 可视化数据
x_train_3d = x_train[:, :3]   # 获取训练集的前三个特征
clf_3d = SVC(kernel='rbf', C=1.0, gamma='scale')  # 创建支持向量机分类器
clf_3d.fit(x_train_3d, y_train)  # 训练模型
plot_decision_boundary(x_train_3d, y_train, clf_3d)  # 可视化数据和模型



