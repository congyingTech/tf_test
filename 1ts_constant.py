# coding=utf-8
"""
@author: congying
@email: wangcongyinga@gmail.com 
@time: 2019/2/19
"""

# tensorflow的可视化
from __future__ import print_function
import tensorflow as tf
import os

# flags是属于全局的常量，通常用来定义一些超参啊之类的
tf.app.flags.DEFINE_string('log_dir', os.path.dirname(os.path.abspath(__file__)) + '/logs',
                           'Directory where event logs are written to.')
FLAGS = tf.app.flags.FLAGS
if not os.path.isabs(os.path.expanduser(FLAGS.log_dir)):
    raise ValueError('You must assign absolute path for --log_dir')

# Defining some constant values
a = tf.constant(5.0, name="a")
b = tf.constant(10.0, name="b")

# Some basic operations
x = tf.add(a, b, name="add")
y = tf.div(a, b, name="divide")

# Run the session
with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)
    print("output: ", sess.run([a, b, x, y]))

# Closing the writer.
writer.close()
sess.close()
