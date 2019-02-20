# coding=utf-8
"""
@author: congying
@email: wangcongyinga@gmail.com 
@time: 2019/2/19
"""
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", reshape=True, one_hot=False)

# name_scope
# 简单来说name_scope是给Op_name加前缀的,variable_scope是给变量variable_name和Op_name加前缀的.
# 作用域在使用Tensorboard对Graph对象进行可视化的时候很有帮助,作用域会把一些Op划分到较大的语句块当中.
# 使用tensorboard可视化数据流图的时候,每个作用域都对自己的Op进行封装,从而获得更好的可视化效果

# reduce_mean
# 主要用作降维或者计算图像(tensor)的平均值

# Data flags
max_num_checkpoint = 10
num_classes = 2  # LR是二分类问题
batch_size = 512
num_epochs = 10

# Learning rate flags
initial_learning_rate = 0.001
learning_rate_decay_factor = 0.95
num_epochs_per_decay = 1  # 一次decay只训练一轮epoch

# status flags
is_training = False
fine_tuning = False
online_test = True
allow_soft_placement = True
log_device_placement = False

########################
### Data Processing ####
########################
# Organize the data and feed it to associated dictionaries.

data = {}
data['train/image'] = mnist.train.images
data['train/label'] = mnist.train.labels
data['test/image'] = mnist.test.images
data['test/label'] = mnist.test.labels

index_list_train = []
n = data['train/label'].shape[0]
for sample_index in range(n):
    label = data['train/label'][sample_index]
    if label == 0 or label == 1:  # 洗数据，把label是0或者1的洗出来，不是的则抛弃
        index_list_train.append(sample_index)

# 根据洗出来的有效index，把有效数据弄出来
data['train/image'] = mnist.train.images[index_list_train]
data['train/label'] = mnist.train.labels[index_list_train]

# 同理把test的有效数据洗出来
index_list_test = []
n = data['test/label'].shape[0]
for sample_index in range(n):
    label = data['test/label'][sample_index]
    if label == 0 or label == 1:
        index_list_test.append(sample_index)

data['test/image'] = mnist.test.images[index_list_test]
data['test/label'] = mnist.test.labels[index_list_test]


train_dim = data['train/image'].shape
nums_train_data = train_dim[0]
train_feature = train_dim[1]

graph = tf.Graph()
with graph.as_default():
    global_step = tf.Variable(0, name='global_step', trainable=False)
    decay_steps = int(nums_train_data / batch_size * num_epochs_per_decay)  # learning rate 指数衰减的步数
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps,
                                               learning_rate_decay_factor, staircase=True,
                                               name='exponential_decay_learning_rate')
    # define placeholders
    image_place = tf.placeholder(tf.float32, shape=([None, train_feature]), name='image')
    label_place = tf.placeholder(tf.int32, shape=([None, ]), name='gt')
    label_one_hot = tf.one_hot(label_place, depth=num_classes, axis=-1)
    dropout_param = tf.placeholder(tf.float32)

    # model + loss + accuracy
    logits = tf.contrib.layers.fully_connected(inputs=image_place, num_outputs=num_classes, scope='fc')
    # 定义loss
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_one_hot))
    # 定义准确率
    with tf.name_scope('accuracy'):
        predict_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(label_one_hot, 1))
        accuracy = tf.reduce_mean(tf.cast(predict_correct, tf.float32))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    with tf.name_scope('train_op'):
        gradients_and_variables = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(gradients_and_variables, global_step=global_step)

    # session的配置
    session_conf = tf.ConfigProto(
        allow_soft_placement=allow_soft_placement,
        log_device_placement=log_device_placement
    )
    # 自定义session
    sess = tf.Session(graph=graph, config=session_conf)
    with sess.as_default():
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        checkpoint_path = 'models'
        checkpoint_prefix = 'model'
        if fine_tuning:
            saver.restore(sess, os.path.join(checkpoint_path, checkpoint_prefix))
            print("Model restored for fine-tuning...")
