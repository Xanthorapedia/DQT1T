import tensorflow as tf
from dqn import agent, game
import pickle
import os


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    agent = agent.Agent(tf.Session())
    game = game.Game()

    agent.cur_state = game.get_state()

    for episode in range(int(1e12)):
        action = agent.act()
        # new_state = game.act(action)
        # agent.observe(new_state)
        # if new_state.terminal:
        #     print('gg, score: {}'.format(game.prev_score))
        #     game.reset()
        #     agent.cur_state = game.get_state()
