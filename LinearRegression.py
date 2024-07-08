import numpy as np  # 导入需要用到的package
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D

# 读取数据集
data_path = 'D:/数据分析/dataset/boston_house_price/housing_data.csv'  # 数据集路径
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM','AGE','DIS', 'RAD', 'TAX',
                 'PTRATIO', 'B', 'LSTAT', 'MEDV']  # 特征名称
feature_num = len(feature_names)  # 数据集列数
# print(feature_num)  # 输出数据集列数
data_frame = pd.read_csv(data_path, delimiter=',', names=feature_names)  # 读取数据集
data = data_frame.to_numpy()  # 转换为numpy数组
data = data[1:, :]  # 删除第一行
data = data.astype(float)

# print(data.shape)  # 输出数组形状
# print(data)  # 输出数组

# 划分训练集和测试集
random.shuffle(data)  # 随机打乱数据
train_data = data[:int(data.shape[0]*0.8), :]  # 取前80%的数据作为训练集
test_data = data[int(data.shape[0]*0.8):, :]  # 取后20%的数据作为测试集

# 归一化
maximums, minimums = train_data.max(axis=0), train_data.min(axis=0)  # 计算每列的最大值和最小值
avgs = train_data.sum(axis=0) / train_data.shape[0]  # 计算每列的平均值
for i in range(feature_num):
    train_data[:, i] = (train_data[:, i] - minimums[i]) / (maximums[i] - minimums[i])
    test_data[:, i] = (test_data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

# print(train_data[0])


# 定义模型
class Model(object):
    def __init__(self, num_of_weights):
        # 初始化权重
        # self.weights = np.zeros(num_of_weights, 1)  # 初始化权重为0
        self.w = np.random.randn(num_of_weights, 1)  # 初始化权重为随机数
        self.b = 0  # 偏置

    def forward(self, x):
        # 前向传播
        z = np.dot(x, self.w) + self.b  # 计算预测值
        return z

    def loss(self, x, y):
        # 损失函数
        z = self.forward(x)  # 计算预测值
        loss = np.mean((z - y) ** 2)  # 均方误差
        return loss

    def backward(self, x, y):
        # 反向传播
        z = self.forward(x)  # 计算预测值
        gradient_w = np.dot(x.T, (z - y)) / x.shape[0]  # 计算权重梯度
        gradient_b = np.mean((z - y))  # 计算偏置梯度
        return gradient_w, gradient_b

    def update(self, gradient_w, gradient_b, learning_rate):
        # 更新权重
        self.w -= learning_rate * gradient_w  # 更新权重
        self.b -= learning_rate * gradient_b  # 更新偏置

    def train(self, x, y, learning_rate, num_of_epochs):
        # 训练模型
        losses = []
        for epoch in range(num_of_epochs):
            # z = self.forward(x)  # 计算预测值
            loss = self.loss(x, y)  # 计算损失函数
            gradient_w, gradient_b = self.backward(x, y)  # 计算梯度
            self.update(gradient_w, gradient_b, learning_rate)  # 更新权重
            losses.append(loss)
            # print(loss)
            if (epoch+1) % 100000 == 0:
                print('Epoch: {}, Loss: {}'.format(epoch+1, loss))

        return losses

    def test(self, x, y):
        # 测试模型
        # z = self.forward(x)  # 计算预测值
        loss = self.loss(x, y)  # 计算损失函数
        print('Test Loss: {}'.format(loss))


if __name__ == '__main__':
    net = Model(feature_num-1)  # 创建模型
    x = train_data[:, :-1]  # 特征值
    y = train_data[:, -1:]  # 目标值
    epochs = 10000000  # 迭代次数
    learning_rate = 0.001  # 学习率
    losses = net.train(x, y, learning_rate=learning_rate, num_of_epochs=epochs)  # 训练模型
    plot_x = np.arange(epochs)  # 画图的x轴数据
    plot_y = np.array(losses)  # 画图的y轴数据
    plt.plot(plot_x, plot_y)  # 画图
    plt.show()

    # 模型评估
    print("回归方程系数：", net.w)
    print("回归方程偏置：", net.b)
    y_pred = net.forward(test_data[:, :-1])  # 预测测试集
    # print('y_pred: {}'.format(y_pred))
    y_mean = np.mean(test_data[:, -1:])  # 计算平均值
    MSE = metrics.mean_squared_error(test_data[:, -1:], y_pred)  # 计算均方误差
    # print('MSE: {}'.format(MSE))
    RMSE = np.sqrt(MSE)  # 计算均方根误差
    # print('RMSE: {}'.format(RMSE))
    # 计算R平方
    # r2 = metrics.r2_score(test_data[:, -1:], y_pred)
    # print(r2)
    sse = np.sum((y_pred - test_data[:, -1:]) ** 2)  # 预测值与真实值之间的平方误差
    print(sse)
    sst = np.sum((test_data[:, -1:] - y_mean) ** 2)  # 真实值与平均值之间的平方误差
    print(sst)
    r_square = 1 - sse / sst  # R平方
    print('R square: {}'.format(r_square))
    plt.scatter(test_data[:, -1:], test_data[:, -1:], label="real-date")  # 画图
    plt.scatter(test_data[:, -1:], y_pred, label="predict-date")  # 画图
    plt.legend()  # 显示图例
    plt.show()


# net = Model(feature_num-1)  # 创建模型
# x = train_data[:, :-1]  # 特征值
# y = train_data[:, -1:]  # 目标值
# # print(x[0])
# # print(y[0])
# z = net.forward(x[0])  # 计算预测值
# print(z, y[0])
# loss = net.loss(x[0:3], y[0:3])  # 计算损失函数
# print(loss)

# losses = []
# w5 = np.arange(-160.0, 160.0, 1)
# w9 = np.arange(-160.0, 160.0, 1)
# losses = np.zeros([len(w5), len(w9)])
#
# # 计算设定区域内每个参数对应的损失函数
# for i in range(len(w5)):
#     for j in range(len(w9)):
#         net.w[5] = w5[i]
#         net.w[9] = w9[j]
#         z = net.forward(x)  # 计算预测值
#         loss = net.loss(x, y)  # 计算损失函数
#         losses[i, j] = loss/1000

# print(losses.shape)
# 画图
# fig = plt.figure()
# # ax = Axes3D(fig)
# ax = fig.add_subplot(111, projection='3d')  # 创建一个3D图形
#
# w5, w9 = np.meshgrid(w5, w9)  # 生成网格点坐标矩阵
# # 绘制3D图，rstride（row）指定行的跨度，设置颜色映射
# ax.plot_surface(w5, w9, losses, rstride=1, cstride=1, cmap='rainbow')  # 绘制曲面图
# ax.set_xlabel('w5')  # 设置x轴标签
# ax.set_ylabel('w9')  # 设置y轴标签
# ax.set_zlabel('loss')  # 设置z轴标签
# plt.show()

# # 梯度计算
# x1 = x[0]
# y1 = y[0]
# z1 = net.forward(x1)
# print('x1:{}, shape:{}'.format(x1, x1.shape))
# print('y1:{}, shape:{}'.format(y1, y1.shape))
# print('z1:{}, shape:{}'.format(z1, z1.shape))
# gradient_w = 2 * (z1 - y1) * x1
# print('gradient_w:{}'.format(gradient_w))
