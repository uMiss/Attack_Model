import tensorflow as tf
from model import Attack_model
from Data_preprocess import PARAMETER_FILE

class inference:
    def __init__(self):
        self.attack_model = Attack_model()
        self.parameter_path = PARAMETER_FILE
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def predict(self, x_t_data1, x_t_data2, y_t_data):
        self.saver.restore(self.sess, self.parameter_path)
        feed = {self.attack_model.x1: x_t_data1,
                self.attack_model.x2: x_t_data2,
                self.attack_model.ys: y_t_data}
        predition = self.sess.run(self.attack_model.pred_digits, feed_dict=feed)
        test_acc = self.sess.run(self.attack_model.accuracy, feed_dict=feed)
        return predition, test_acc