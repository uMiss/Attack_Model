import numpy as np
from Inference import inference

# 在随机数上测试准确率
def main():
    # ................................产生随机数.................................
    ram_x_t_data1 = np.random.rand(50, 10)
    ram_x_t_data2 = np.random.rand(50, 256)
    ram_y_t_data = np.ones([500, 1])

    path = './vgg_output/Data1_fc2_D_no_D1.txt'
    path1 = './vgg_output/Data1_fc1_D_no_D1.txt'
    input = np.loadtxt(path)
    input1 = np.loadtxt(path1)

    Inter = inference()
    result, accuracy = Inter.predict(input, input1, ram_y_t_data)
    print('The result of predicting is \n', result)
    print('accuracy is ', accuracy)

if __name__ == '__main__':
    main()


