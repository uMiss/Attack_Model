import tensorflow as tf
import Data_preprocess
from Data_preprocess import *

class Attack_model:
    def __init__(self):
        self.x1, self.x2 = Data_preprocess.pre_placeholder(input[0].size, input1[0].size)
        self.ys = tf.placeholder(tf.float32, [None, 1])
        self.pred_digits = self.construct_net()
        # 输出结果进行处理
        self.pred_encoder = tf.round(self.pred_digits)
        # ................................................定义 loss 表达式..............................................
        self.cost = tf.reduce_mean(tf.reduce_sum(tf.square(self.ys - self.pred_digits), reduction_indices=[1]))
        # 训练的优化策略，一般有梯度下降（GradientDescentOptimizer）、AdamOptimizer等。minimize(loss)是让 loss 达到最小
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
        # tf.argmax()返回的是vector中的最大值的索引号
        # tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素
        self.correct_pred = tf.equal(self.pred_encoder, self.ys)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def construct_net(self):
        # output1_fc1
        output1_fc1 = tf.contrib.layers.fully_connected(self.x1, 128)
        # output_fc2
        output1_fc2 = tf.contrib.layers.fully_connected(output1_fc1, 64)

        # output2_fc1
        output2_fc1 = tf.contrib.layers.fully_connected(self.x2, 128)
        # output_fc2
        output2_fc2 = tf.contrib.layers.fully_connected(output2_fc1, 64)

        # 拼接两个tensor
        encoder_input = tf.concat([output1_fc2, output2_fc2], 1)

        # encoder_fc1
        encoder_fc1 = tf.contrib.layers.fully_connected(encoder_input, 256)
        # encoder_fc2
        encoder_fc2 = tf.contrib.layers.fully_connected(encoder_fc1, 128)
        # encoder_fc3
        encoder_fc3 = tf.contrib.layers.fully_connected(encoder_fc2, 64)
        # encoder_prediction
        encoder_fc4 = tf.contrib.layers.fully_connected(encoder_fc3, 1)

        return encoder_fc4