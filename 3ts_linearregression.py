# coding=utf-8
"""
@author: congying
@email: wangcongyinga@gmail.com 
@time: 2019/2/19
"""
import tensorflow as tf
import numpy as np
import xlrd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

DATA_FILE = "data/fire_theft.xls"

book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
num_samples = sheet.nrows - 1

#######################
## Defining flags —— 用于接受命令行传递参数，相当于接受argv。首先调用自带的DEFINE_string，DEFINE_boolean DEFINE_integer, DEFINE_float设置不同类型的命令行参数及其默认值。当然，也可以在终端用命令行参数修改这些默认值。
#####
## placeholder是Tensorflow的占位符节点，由placeholder创建，也是一种常量。是由用户在调用run方法时传递的，也可以将placeholder理解为一种形参。
#######################

# 定义训练的轮数——根据输入定义
tf.app.flags.DEFINE_integer('num_epochs', 50, 'The number of epochs for training the model. Default=50')
FLAGS = tf.app.flags.FLAGS
W = tf.Variable(0.0, name="weights")
b = tf.Variable(0.0, name='bias')


def inputs():
    X = tf.placeholder(tf.float32, name="X")
    Y = tf.placeholder(tf.float32, name="Y")
    return X, Y


def inference(X):
    return W*X + b


def loss(X, Y):
    """
    平方损失函数
    :param X:
    :param Y:
    :return:
    """
    Y_predicted = inference(X)
    return tf.squared_difference(Y, Y_predicted)


def train(loss):
    learning_rate = 0.0001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


# 初始化gif背景
fig, ax = plt.subplots()
fig.set_tight_layout(True)
xdata = data[:, 0]
ydata = data[:, 1]
y_predict = []
ax.scatter(xdata, ydata)
line, = ax.plot([], [])  # 初始化line


# evaluate and plot
def gen_gif(epoch_num):
    # 定义gif的基本要素
    label = 'epoch:%d - Predicted' % epoch_num
    line.set_xdata(xdata)
    line.set_ydata(y_predict[epoch_num])
    ax.set_xlabel(label)
    # plt.plot(input_values, labels, 'ro', label='Original')
    # plt.plot(input_values, predict_value, label='epoch:%d - Predicted' % epoch_num)
    return line, ax


with tf.Session() as sess:
    # 初始化所有变量，包括W和b
    sess.run(tf.global_variables_initializer())
    X, Y = inputs()

    # 创建train的operation
    train_loss = loss(X, Y)
    train_op = train(train_loss)
    images = []
    for epoch_num in range(FLAGS.num_epochs):
        loss_value = 0.0
        for x, y in data:
            # 将X，Y两个占位符传入到train_op和train loss里面
            loss_value, _ = sess.run([train_loss, train_op], feed_dict={X: x, Y: y})

        print('epoch %d, loss=%f' % (epoch_num + 1, loss_value))
        wcoeff, bias = sess.run([W, b])
        print('epoch %d, wcoeff-%d, bias-%d' % (epoch_num+1, wcoeff, bias))
        predict_value = data[:, 0] * wcoeff + bias
        y_predict.append(predict_value)

# 绘制gif动图
anim = FuncAnimation(fig, gen_gif, frames=np.arange(0, FLAGS.num_epochs), interval=200)
plt.show()
anim.save('images/3ts.gif', dpi=80, writer='imagemagick')




