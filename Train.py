from model import Attack_model
import numpy as np
from Data_preprocess import *
import tensorflow as tf

def main():
    # ..........................................训练..........................................
    x_data1, x_data2, y_data, x_v_data1, x_v_data2, y_v_data, x_t_data1, x_t_data2, y_t_data = pre_data(input, input1, label)
    model = Attack_model()
    # 初始化所有变量
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # 激活会话
    iteration = 0
    with tf.Session() as sess:
        sess.run(init)
        for e in range(100):
            # feed = {model.x1: x_data1, model.x2: x_data2, model.ys: y_data}
            loss, _ = sess.run([model.cost, model.optimizer], feed_dict={model.x1: x_data1, model.x2: x_data2, model.ys: y_data})
            print("Epoch: {}/{}".format(e + 1, 100),
                  "Iteration: {}".format(iteration),
                  "Training loss: {:.5f}".format(loss))
            iteration += 1

            if iteration % 5 == 0:
                feed = {model.x1: x_v_data1,
                        model.x2: x_v_data2,
                        model.ys: y_v_data}
                val_acc = sess.run(model.accuracy, feed_dict=feed)
                print("Epoch: {}/{}".format(e, 100),
                      "Iteration: {}".format(iteration),
                      "Validation Acc: {:.4f}".format(val_acc))
        saver.save(sess, "checkpoints/attack_model.ckpt")
        feed = {model.x1: x_t_data1,
                model.x2: x_t_data2,
                model.ys: y_t_data}
        test_acc = sess.run(model.accuracy, feed_dict=feed)
        print(np.array(sess.run(model.pred_digits, feed_dict=feed)))
        print("Test accuracy: {:.4f}".format(test_acc))

if __name__ == '__main__':
    main()

