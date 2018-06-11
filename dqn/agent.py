import tensorflow as tf
import numpy as np
import os
from dqn import memory
from dqn.utils import misc
from dqn.utils.bean import Sample
from dqn.acddpg import ActorCritic


class Agent:
    debug = False
    with_tb = True
    LOG_DIR = './train_logs'

    def __init__(self, session):
        self.on_sample = True
        self.MAX_TIME = 1000
        self.MIN_TIME = 200
        self.memory = memory.Memory()
        self.batch_size = 48
        self.cur_state = None
        self.action_taken = None
        self.epsilon = 0.8  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.training_freq = 5
        self.step_per_episode = 1000
        self.update_freq = 10
        self.saving_freq = 100
        self.training_start = 3
        self.training_episode = 0
        self.episode_step = 0
        self.episode_step_t = tf.placeholder(tf.int64, name='step_in_episode')
        self.sess = session
        self.model = ActorCritic(session)

        # load model
        if not os.path.isdir(self.LOG_DIR):
            os.mkdir(self.LOG_DIR)
        checkpoint = tf.train.get_checkpoint_state(self.LOG_DIR)
        with self.sess.graph.as_default():
            # print([n.net_name for n in tf.get_default_graph().as_graph_def().node])
            if checkpoint and checkpoint.model_checkpoint_path:
                self.model.restore(checkpoint.model_checkpoint_path)
                print('Model successfully restored')
            else:
                self.model.save(self.LOG_DIR, self.training_episode * self.step_per_episode + self.episode_step)
                print('New model created')

        # launch tensorboard
        if self.with_tb:
            os.popen('tensorboard --logdir=%s' % self.LOG_DIR)

    def predict(self):
        return self.model.actor(self.cur_state)

    def observe(self, sample: Sample):
        sample.state.img = misc.preprocess(sample.state.img)
        if self.cur_state is not None:
            sample.p_state = self.cur_state
            sample.action = self.action_taken
            self.memory.memorize(sample)
        self.cur_state = sample.state  # act() takes place after this step on cur_state
        if sample.terminal:
            self.cur_state = None

    def act(self):
        if not self.on_sample:
            # explore
            if self.episode_step < self.training_start or np.random.rand() <= self.epsilon:
                self.action_taken = np.random.randint(*self.model.actor.out_range)
                print('rand:', self.action_taken)
            # exploit
            else:
                self.action_taken = int(self.model.predict_action(self.cur_state))
                print('agent:', self.action_taken)

        self.episode_step += 1

        if self.episode_step % 40 == 0:
            print('episode_step: ', self.episode_step)

        if self.episode_step >= self.training_start and self.episode_step % self.training_freq == 0:
            # decay epsilon
            self.epsilon *= 1 if self.epsilon <= self.epsilon_min else self.epsilon_decay
            losses = []
            for _ in range(1):
                self.debug = True
                losses.append(self.replay(self.batch_size))
            print("episode: {}, step: {}, ave_loss: {}".format(
                self.training_episode, self.episode_step, np.mean(np.abs(losses))))

        if self.episode_step % self.saving_freq == 0:
            print('saving model')
            self.model.save(self.LOG_DIR, self.training_episode * self.step_per_episode + self.episode_step)

        # update target network (copy from action network)
        if self.episode_step % self.update_freq == 0:
            self.memory.prob_tree.save()  # TODO: move this elsewhere
            print('update target')
            self.model.update_target()

        if self.episode_step % self.step_per_episode == 0:
            self.training_episode += 1
            self.episode_step = 0
            self.epsilon = 0.05

        return self.action_taken

    # sample a minibatch to train
    def replay(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        samples = self.memory.unpacked_sample(batch_size)
        losses = self.model.train(samples)
        self.memory.update_prob(losses, samples['refs'])
        return np.mean(losses)
