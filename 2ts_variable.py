# coding=utf-8
"""
@author: congying
@email: wangcongyinga@gmail.com 
@time: 2019/2/19
"""

from __future__ import print_function
import tensorflow as tf
import os
from tensorflow.python.framework import ops

tf.app.flags.DEFINE_string('log_dir', os.path.dirname(os.path.abspath(__file__)) + '/logs',
                           'Directory where event logs are written to.')
FLAGS = tf.app.flags.FLAGS
if not os.path.isabs(os.path.expanduser(FLAGS.log_dir)):
    raise ValueError('You must assign absolute path for --log_dir')

# 存储变量的tensor需要初始化
# Create three variables with some default values.
weights = tf.Variable(tf.random_normal([2, 3], stddev=0.1), name='weights')  # <tf.Variable 'weights:0' shape=(2, 3) dtype=float32_ref>

bias = tf.Variable(tf.zeros([3]), name='biases')
custom_variable = tf.Variable(tf.zeros([3]), name="custom")  # Tensor("zeros:0", shape=(3,), dtype=float32)

# 把三个tensor存储到一个list里面
all_variables_list = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

# 第一种初始化
# 自定义初始化list的变量元素
variable_list_custom = [weights, custom_variable]

init_custom_op = tf.variables_initializer(var_list=variable_list_custom)

# 第二种初始化
# 全局初始化-方案1
init_all_variable1 = tf.variables_initializer(var_list=all_variables_list)

# 全局初始化-方案2
init_all_variable2 = tf.global_variables_initializer()

# 第三种初始化
# 使用其他现有变量初始化变量
WeightsNew = tf.Variable(weights.initialized_value(), name="WeightsNew")  # 先把现有的变量weights放到新创建的变量里面
init_WeightsNew_op = tf.variables_initializer(var_list=[WeightsNew])  # 然后选择的初始化


# 一个graph 有operation和tensor两种元素，分别对应着图的节点和边
# 一般一个session都会对应一张默认的图，但是可以自定义多个图，但是一般没有这个必要
with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)

    print(sess.run(init_custom_op))
    print(sess.run(init_all_variable1))
    print(sess.run(init_WeightsNew_op))








