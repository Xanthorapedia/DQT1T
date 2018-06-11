import tensorflow as tf


class BaseNet:

    IMG_SIZE = 224
    IN_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3]
    initializer_w = tf.truncated_normal_initializer(0, 0.05)
    initializer_b = tf.truncated_normal_initializer(1, 0.05)

    def __init__(self, sess):
        self.learning_rate = 1e-4
        self.sess = sess
        self.loss = self.losses = self.optimizer = None

    @staticmethod
    def make_conv(net_name, layer_name, feed, ks, stride, is_training, bn=True,
                  activation=tf.nn.leaky_relu):
        with tf.variable_scope('%s-%s_conv' % (net_name, layer_name)):
            w = tf.get_variable('%s-w' % layer_name, shape=ks, initializer=BaseNet.initializer_w)
            b = tf.get_variable('%s-b' % layer_name, shape=[ks[-1]], initializer=BaseNet.initializer_b)
            out = tf.nn.conv2d(feed, w, strides=[1, stride, stride, 1], padding='SAME', name='%s-conv' % layer_name)
            out = tf.nn.bias_add(out, b, name='%s-biad_add' % layer_name)
            if bn:
                out = tf.layers.batch_normalization(out, name='%s-bn' % layer_name, training=is_training, renorm=True)
            out = activation(out, name='%s-activation' % layer_name)
        return out

    @staticmethod
    def make_fc(net_name, layer_name, feed, ks, keep_prob=tf.constant(value=1, dtype=tf.float32),
                activation=tf.nn.leaky_relu):
        with tf.variable_scope('%s-%s_fc' % (net_name, layer_name)):
            w = tf.get_variable('%s-w' % layer_name, shape=ks, initializer=BaseNet.initializer_w)
            b = tf.get_variable('%s-b' % layer_name, shape=[ks[-1]], initializer=BaseNet.initializer_b)
            out = tf.matmul(feed, w, name='%s-mat' % layer_name)
            out = tf.nn.bias_add(out, b, name='%s-bias_add' % layer_name)
            out = tf.nn.dropout(out, keep_prob, name='%s-drop' % layer_name)
            out = activation(out, name='%s-activation' % layer_name)
        return out
