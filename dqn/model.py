# Adapted from and inspired by Richard-An
# https://github.com/Prinsphield/Wechat_AutoJump/blob/master/cnn_coarse_to_fine/config/base.fine/model.py
from functools import reduce
from operator import mul
import tensorflow as tf
import numpy as np


def conv2d(name, input, ks, stride):
    with tf.name_scope(name):
        with tf.variable_scope('%s-network' % name):
            w = tf.get_variable('%s-w' % name, shape=ks, initializer=tf.truncated_normal_initializer(0, 0.01))
            b = tf.get_variable('%s-b' % name, shape=[ks[-1]], initializer=tf.constant_initializer(1))
            out = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding='SAME', name='%s-conv' % name)
            out = tf.nn.bias_add(out, b, name='%s-biad_add' % name)
    return out


def make_conv_bn_relu(name, input, ks, stride, is_training):
    out = conv2d('%s-conv' % name, input, ks, stride)
    out = tf.layers.batch_normalization(out, name='%s-bn' % name, training=is_training)
    out = tf.nn.relu(out, name='%s-relu' % name)
    return out


def make_fc(name, input, ks, keep_prob=tf.constant(value=1, dtype=tf.float32), activation=True):
    with tf.name_scope(name):
        with tf.variable_scope('%s-fc' % name):
            w = tf.get_variable('%s-w' % name, shape=ks, initializer=tf.truncated_normal_initializer(0, 0.01))
            b = tf.get_variable('%s-b' % name, shape=[ks[-1]], initializer=tf.constant_initializer(1))
            output = tf.matmul(input, w, name='%s-mat' % name)
            output = tf.nn.bias_add(output, b, name='%s-bias_add' % name)
            output = tf.nn.dropout(output, keep_prob, name='%s-drop' % name)
            if activation:
                output = tf.nn.relu(output, name='%s-relu' % name)
    return output


class Model:
    input_channel = 3
    img_shape = (240, 240, input_channel)
    batch_size = 8
    output_size = 4

    def __init__(self, name, session, img_holder=None, time_holder=None, step_holder=None,
                 training=None, target=None, action_taken=None, keep_prob=None,
                 learning_step=None, has_optimizer=False):
        self.action_one_hot = None
        self.q_val = None

        self.name = name
        self.sess = session
        # input to net
        self.img = tf.placeholder(np.float32, shape=np.append([None], self.img_shape), name=name + '-in_img') \
            if img_holder is None else img_holder
        self.time = tf.placeholder(np.float32, shape=[None, 1], name=name + '-in_time') \
            if time_holder is None else time_holder
        self.step = tf.placeholder(np.float32, shape=[None, 1], name=name + '-in_step') \
            if step_holder is None else step_holder
        self.training = tf.placeholder(np.bool, name='is_training') \
            if training is None else training
        self.target = tf.placeholder(np.float32, shape=[None], name=name + '-out_target') \
            if target is None else target
        self.action_taken = tf.placeholder(np.int64, shape=[None], name=name + '-in_action_taken') \
            if action_taken is None else action_taken
        self.keep_prob = tf.placeholder(np.float32, shape=[None], name=name + '-keep_prob') \
            if keep_prob is None else keep_prob
        # input to net
        self.learning_rate = 2e-4
        self.learning_rate_decay = 0.95
        self.learning_rate_min = 1e-5
        self.learning_rate_steps = 1000  # decay every this periods
        self.learning_rate_global_step = tf.placeholder(np.int64, name=name + '-learning_step') \
            if learning_step is None else learning_step
        self.losses = None
        self.out, self.loss, self.optimizer, self.saver, _ = self.make(has_optimizer)

    def make(self, has_optimizer):
        name = self.name
        with tf.name_scope(name):
            with tf.variable_scope(name):
                # network
                output = conv2d('conv1', self.img, [3, 3, self.input_channel, 16], 2)
                # out = tf.layers.batch_normalization(out, net_name='bn1', training=is_training)
                output = tf.nn.relu(output, name='relu1')

                output = make_conv_bn_relu('conv2', output, [5, 5, 16, 64], 1, self.training)
                output = tf.nn.max_pool(output, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

                output = make_conv_bn_relu('conv3', output, [7, 7, 64, 128], 1, self.training)
                output = tf.nn.max_pool(output, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

                output = make_conv_bn_relu('conv4', output, [9, 9, 128, 256], 1, self.training)
                output = tf.nn.max_pool(output, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

                # output = make_conv_bn_relu('conv5', output, [9, 9, 256, 512], 1, self.training)
                # output = tf.nn.max_pool(output, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
                shape = output.get_shape().as_list()[1:]
                length = reduce(mul, shape)

                output = tf.reshape(output, [-1, length])
                output = tf.concat([output, self.time, self.step], axis=1)  # add time and step here

                # print(output.get_shape().as_list())

                output = make_fc('fc1', output, [length + 2, 256], keep_prob=self.keep_prob)
                output = make_fc('fc2', output, [256, 128], keep_prob=self.keep_prob)
                output = make_fc('fc3', output, [128, self.output_size], activation=False)

                # output = tf.placeholder(np.float32, shape=[2, 1], net_name=net_name + '-lll')

            # loss and optimizer
            with tf.variable_scope('%s-optimizer' % name):
                self.action_one_hot = tf.one_hot(self.action_taken, self.output_size, 1.0, 0.0, name='action_one_hot')
                self.q_val = tf.reduce_sum(output * self.action_one_hot, reduction_indices=1, name='q_val')  # Q*
                # loss = tf.reduce_mean(self.target - self.q_val, net_name='loss')
                loss = tf.losses.mean_squared_error(self.target, self.q_val)
                self.losses = self.target - self.q_val

                if has_optimizer:
                    learning_rate = tf.maximum(tf.constant(self.learning_rate_min),
                        tf.train.exponential_decay(
                            tf.constant(self.learning_rate),
                            self.learning_rate_global_step,
                            tf.constant(self.learning_rate_steps),
                            tf.constant(self.learning_rate_decay),
                            staircase=True
                        )
                                               )
                    optimized = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
                    saver = tf.train.Saver(max_to_keep=10)
                    summary = tf.summary.scalar('loss', loss)
                    summary_writer = tf.summary.FileWriter('./logs', self.sess.graph)
                else:
                    optimized = None
                    saver = None
                    summary_writer = None

        self.sess.run(tf.global_variables_initializer())
        return output, loss, optimized, saver, summary_writer

    def eval(self, state, training=False):
        with self.sess.as_default():
            out = self.out.eval({
                self.img: [state.img],
                self.time: [[state.time]],
                self.step: [[state.step]],
                self.training: training,
                self.keep_prob: 1
            })
        return out, tf.argmax(self.out, axis=1), self.out  # originally 1 * n array[[a, b]], turns into list


if __name__ == '__main__':
    pass
