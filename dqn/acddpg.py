import tensorflow as tf
import numpy as np
import time
from dqn.network.base import BaseNet
from dqn.network.critic import CriticNet
from dqn.network.actor import ActorNet


class ActorCritic:
    TAU = 0.03

    def __init__(self, sess):
        # param
        self.gamma = 0.95  # reward discount
        self.learning_rate = 1e-3
        # setup
        self.sess = sess
        self.itr = 0  # TODO: remove this guy
        with tf.name_scope('networks'):
            self.state = tf.placeholder(tf.float32, shape=BaseNet.IN_SHAPE, name='s_in')
            self.action = tf.placeholder(tf.float32, shape=[None, 1], name='a_in')
            self.actor = ActorNet('actor', sess, True, self.state)
            self.critic = CriticNet('critic', sess, True, self.state, self.action)
            self.t_actor = ActorNet('t_actor', sess, False, self.state)
            self.t_critic = CriticNet('t_critic', sess, False, self.state, self.t_actor.output)
        # training
        with tf.name_scope('update_target/'):  # make name_scope unique
            vs = tf.trainable_variables()
            self.critic_net_w = [w for w in vs if w.name.startswith('critic')]
            self.t_critic_net_w = [w for w in vs if w.name.startswith('t_critic')]
            self.actor_net_w = [w for w in vs if w.name.startswith('actor')]
            self.t_actor_net_w = [w for w in vs if w.name.startswith('t_actor')]
            self.update_actor_target = tf.group(*[tf.assign(tw, aw * self.TAU + tw * (1 - self.TAU),
                                                  name='%s_from_%s' % (self._tensor_name(tw), self._tensor_name(aw)))
                                        for tw, aw in zip(self.t_actor_net_w, self.actor_net_w)])
            self.update_critic_target = tf.group(*[tf.assign(tw, cw * self.TAU + tw * (1 - self.TAU),
                                                   name='%s_from_%s' % (self._tensor_name(tw), self._tensor_name(cw)))
                                         for tw, cw in zip(self.t_critic_net_w, self.critic_net_w)])
        self.sess.run(tf.global_variables_initializer())
        with tf.name_scope('summaries/'):
            self.summary_all = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter('./train_logs', graph=self.sess.graph)
        self.saver = tf.train.Saver()

    @staticmethod
    def _tensor_name(tensor):
        return tensor.name[tensor.name.rfind('/') + 1:tensor.name.rfind(':')]

    def update_target(self):
        self.sess.run([self.update_actor_target, self.update_critic_target])

    def predict_action(self, state):
        return self.sess.run(self.actor.output, {self.state: [state.img],
                                                 self.actor.training: False})

    def predict_q(self, state, action):
        return self.sess.run(self.critic.output, {self.state: [state.img], self.action: [action],
                                                  self.critic.training: False})

    @staticmethod
    def _print_stat(name, arr):
        print("{}: mean(a): {}, max: {}, min: {}".format(name, np.abs(np.mean(arr)), np.max(arr), np.min(arr)))

    def train(self, samples):
        # print(self.sess.run([self.t_actor.layers['fc3'], self.t_critic.layers['conv4'], self.t_critic.layers['fc3']], feed_dict={
        #     self.state: samples['states'],
        #     self.t_critic.training: False,
        #     self.t_actor.training: False
        # }))
        # critic:
        pred_q = self.sess.run(self.t_critic.output, feed_dict={
            self.state: samples['states'],
            self.t_critic.training: False,
            self.t_actor.training: False
        })
        target_q = (1 - np.array(samples['terminals'])) * self.gamma * pred_q + np.array(samples['rewards'])
        print('TRAINING')
        self._print_stat('t_output', pred_q)
        self._print_stat('target_q', target_q)

        # train critic:
        summary_str, cr_losses, _, grads, output = self.sess.run(
            [self.summary_all, self.critic.losses, self.critic.optimizer, self.critic.dq_da, self.critic.output],
            feed_dict={
                self.state: samples['p_states'],
                self.action: samples['actions'],
                self.critic.target_q: target_q,
                self.actor.training: False,
                self.critic.training: True
            })
        self._print_stat('output', output)
        self._print_stat('cr_losses', cr_losses)
        # print('mse_loss: {}'.format(loss))
        self._print_stat('grads', grads)

        # train actor:
        self.sess.run([self.actor.optimizer], feed_dict={
            self.state: samples['p_states'],
            self.actor.dQ_da: grads[0].flatten(),
            self.actor.training: True,
            self.critic.training: False,
            self.actor.batch_size: float(samples['size'])
        })
        self.summary_writer.add_summary(summary_str, self.itr)
        self.itr += 1
        return cr_losses[:, 0]

    def save(self, path, episode_step):
        self.saver.save(self.sess, '{}/acddpg{}.bak'.format(path, round(time.time())), episode_step)

    def restore(self, path):
        self.saver.restore(self.sess, path)
