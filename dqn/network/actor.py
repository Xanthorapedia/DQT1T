import tensorflow as tf
from dqn.network.base import BaseNet
from operator import mul
from functools import reduce


class ActorNet(BaseNet):

    def __init__(self, name, sess, has_opt, state_in=None):
        super().__init__(sess)
        self.net_name = name
        self.batch_size = tf.placeholder(tf.float32, name=name + '-in_batch_size')
        self.state = tf.placeholder(tf.float32, shape=self.IN_SHAPE, name=name + '-in_state') \
            if state_in is None else state_in
        self.dQ_da = tf.placeholder(tf.float32, shape=[None], name=name + '-in_dq_da')  # from critic
        self.dQ_dw = None
        self.w = None
        self.output = tf.placeholder(tf.float32, shape=[None], name=name + '-out_ac')
        self.keep_prob = tf.constant(1, tf.float32)
        # tf.placeholder(tf.float32, shape=[None], net_name=net_name + '-drop_out_prob')
        self.training = tf.placeholder(tf.bool, name='is_training')
        self.layers = {}
        self.out_range = [100, 1000]
        self.make(has_opt)

    def make(self, has_optimizer):
        net_name = '%s_net' % self.net_name
        # with tf.name_scope('networks/'):
        with tf.variable_scope(net_name):
            # conv layer
            out = self.layers['conv1'] = self.make_conv(net_name, 'conv1', self.state, [3, 3, 3, 16], 1, self.training)
            out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            out = self.layers['conv2'] = self.make_conv(net_name, 'conv2', out, [5, 5, 16, 32], 2, self.training)
            out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            out = self.layers['conv3'] = self.make_conv(net_name, 'conv3', out, [7, 7, 32, 64], 3, self.training)
            out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            out = self.layers['conv4'] = self.make_conv(net_name, 'conv4', out, [9, 9, 64, 128], 4, self.training)
            out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            # fc layer
            shape = out.get_shape().as_list()[1:]
            length = reduce(mul, shape)
            out = tf.reshape(out, [-1, length])
            out = self.layers['fc1'] = self.make_fc(net_name, 'fc1', out, [length, 128], keep_prob=self.keep_prob)
            # out = self.layers['fc2'] = self.make_fc(net_name, 'fc2', out, [256, 128], keep_prob=self.keep_prob)
            out = self.layers['fc2'] = self.make_fc(net_name, 'fc2', out, [128, 1], activation=tf.nn.tanh)
            # map to out_range
            out_a = tf.fill(tf.shape(out), (self.out_range[1] - self.out_range[0]) / 2)
            out_b = tf.fill(tf.shape(out), (self.out_range[1] + self.out_range[0]) / 2)
            out = out * out_a + out_b

            self.output = out
            v = tf.trainable_variables()
            self.w = [w for w in tf.trainable_variables() if w.name.startswith(net_name)]
            if has_optimizer:
                with tf.variable_scope('optimizer'):
                    learning_rate = self.learning_rate
                    self.dQ_dw = tf.gradients(self.output, self.w, grad_ys=-self.dQ_da)# / self.batch_size)
                    self.optimizer = tf.train.AdamOptimizer(learning_rate, name='%s-Adam' % self.net_name)\
                        .apply_gradients(zip(self.dQ_dw, self.w))
