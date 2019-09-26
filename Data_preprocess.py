import tensorflow as tf
import numpy as np
import random

PARAMETER_FILE = "checkpoints/attack_model.ckpt"
LEARNING_RATE = 0.0001

def pre_placeholder(data1_dim, data2_dim):
    x1 = tf.placeholder(tf.float32, [None, data1_dim])
    x2 = tf.placeholder(tf.float32, [None, data2_dim])
    return x1, x2

def pre_data(input, input1, label):
    data_len = len(input1)
    if data_len % 5 == 0:
        num_train = int(data_len * 0.8)
        num_val = int(data_len * 0.1)
        num_test = int(data_len * 0.1)
        x_data1 = input[:num_train]
        x_data2 = input1[:num_train]
        y_data = label[:num_train]

        x_v_data1 = input[num_train:num_train + num_val]
        x_v_data2 = input1[num_train:num_train + num_val]
        y_v_data = label[num_train:num_train + num_val]

        x_t_data1 = input[num_train + num_val:num_train + num_val + num_test]
        x_t_data2 = input1[num_train + num_val:num_train + num_val + num_test]
        y_t_data = label[num_train + num_val:num_train + num_val + num_test]

        return x_data1, x_data2, y_data, x_v_data1, x_v_data2, y_v_data, x_t_data1, x_t_data2, y_t_data

def create_newdatasets():
    path = './vgg_output/data1_fc_2_out.txt'
    path1 = './vgg_output/data1_fc_1_out.txt'
    path2 = './vgg_output/data2_fc_2_out.txt'
    path3 = './vgg_output/data2_fc_1_out.txt'

    Data1_fc2 = np.loadtxt(path)
    Data1_fc1 = np.loadtxt(path1)
    Data2_fc2 = np.loadtxt(path2)
    Data2_fc1 = np.loadtxt(path3)

    # 随机打乱数据
    e = list(zip(Data1_fc2, Data1_fc1))
    random.shuffle(e)
    Data1_fc2[:], Data1_fc1[:] = zip(*e)
    input = Data1_fc2[:300]
    input1 = Data1_fc1[:300]
    # 将矩阵按列合并
    input = np.vstack((input, Data2_fc2))
    input1 = np.vstack((input1, Data2_fc1))

    f = Data1_fc2[300:800]
    g = Data1_fc1[300:800]
    np.savetxt('./vgg_output/Data1_fc2_D_no_D1.txt', f)
    np.savetxt('./vgg_output/Data1_fc1_D_no_D1.txt', g)
    np.savetxt('./vgg_output/Data1_fc2_D_and_D1.txt', input)
    np.savetxt('./vgg_output/Data1_fc1_D_and_D1.txt', input1)
    # print('Data1_fc2.shape is ', Data1_fc2.shape)
    # print('Data1_fc1.shape is ', Data1_fc1.shape)
    # print('Data2_fc2.shape is ', Data2_fc2.shape)
    # print('Data2_fc1.shape is ', Data2_fc1.shape)
    # print('f.shape is ', f.shape)
    # print('g.shape is ', g.shape)
    # print('input.shape is ', input.shape)
    # print('input1.shape is ', input1.shape)

def get_datasets():
    path = './vgg_output/Data1_fc2_D_and_D1.txt'
    path1 = './vgg_output/Data1_fc1_D_and_D1.txt'
    input = np.loadtxt(path)
    input1 = np.loadtxt(path1)
    a = np.ones([300, 1])
    b = np.zeros([200, 1])
    label = np.vstack((a, b))
    temp = list(zip(input, input1, label))
    random.shuffle(temp)
    input[:], input1[:], label[:] = zip(*temp)
    return input, input1, label

input, input1, label = get_datasets()





