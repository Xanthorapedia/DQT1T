import numpy as np
import copy
from dqn.utils.sum_tree import SumTree
from dqn.utils.LRUCache import LRUCache
from dqn.utils.bean import State, Sample
import pickle
import os


class Memory:
    img_dir = './train_mem'  # all (preprocessed) screenshots
    state_dir = img_dir + '/history.pkl'  # everything else except images

    def __init__(self):
        self.img_max_cap = 1000
        self.registered_imgs = LRUCache(self.img_max_cap, save_dir=self.img_dir)
        self.prob_tree = SumTree()
        # self.mem = []
        self.size = 0
        # storing screenshots
        if not os.path.isdir(self.img_dir):
            os.mkdir(self.img_dir)
        self.load_mem()

    def memorize(self, history: Sample):
        self.register(history)
        history_cpy = copy.copy(history)
        history_cpy.state.img = history_cpy.p_state.img = None  # to save space
        # self.mem.append(history_tmp)
        # update prob_tree
        max_prob = 10 if self.prob_tree.size == 0 else self.prob_tree[0].val
        self.prob_tree.set(value=max_prob, data=history_cpy)
        self.size += 1
        # backup
        if self.size % 4 == 0:
            self.save_mem()

    def register(self, game_state: Sample):
        state = game_state.state
        if game_state.terminal:
            state.time_tag = '0'
            p_state = game_state.p_state
            self.registered_imgs.set(p_state.time_tag, p_state.img)
        else:
            self.registered_imgs.set(state.time_tag, state.img)

    def update_prob(self, losses, items):
        for loss, item in zip(losses, items):
            prob = loss * loss + 1.0 / self.prob_tree.size  # at least random
            self.prob_tree.set(prob, idx=item.idx)

    def sample(self, batch_size):
        samples = []
        refs = []
        val = np.random.rand(batch_size) * self.prob_tree[0].sum
        for v in val:
            item, ref = self.get_mem(v)
            while item is None:
                v = np.random.rand(batch_size) * self.prob_tree[0].sum
                item, ref = self.get_mem(v)
            samples.append(item)
            refs.append(ref)
        return samples, refs

    @staticmethod
    def unpack(samples, refs):
        states, p_states = \
            {'imgs': [], 'times': [], 'steps': []}, {'imgs': [], 'times': [], 'steps': []}
        actions, rewards, terminals = [], [], []
        for s in samples:
            states['imgs'].append(s.state.img)
            states['steps'].append([s.state.step])
            states['times'].append([s.state.time])  # convert to 2d array to match dimensions
            p_states['imgs'].append(s.p_state.img)
            p_states['steps'].append([s.p_state.step])
            p_states['times'].append([s.p_state.time])
            actions.append(s.action)
            rewards.append(s.reward)
            terminals.append(s.terminal)
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'p_states': p_states,
            'terminals': terminals,
            'refs': refs,
            'size': len(samples)
        }

    def unpacked_sample(self, batch_size):
        samples, refs = self.sample(batch_size)
        return self.unpack(samples, refs)

    def get_mem(self, val):
        item, index = self.prob_tree.get_item(val)
        sample = copy.copy(item.data)
        # sample.state, sample.p_state = sample.state.copy(), sample.p_state.copy()
        # restore images
        img = self.load_img(sample.state.time_tag)
        p_img = self.load_img(sample.p_state.time_tag)
        if img is None or p_img is None:
            # delete sample
            self.prob_tree.rmv(index)
            print('img not found for sample [%d]:' % index)
            print(str(sample))
            print('sample deleted')
            self.size = self.prob_tree.size
            return None, None
        else:
            sample.state.img = img
            sample.p_state.img = p_img
        return sample, item

    def load_img(self, time_tag):
        return self.registered_imgs.get(time_tag)

    def save_mem(self):
        file = open(self.state_dir, 'wb')
        pickle.dump(self.prob_tree, file)
        self.registered_imgs.save()
        file.close()

    def load_mem(self):
        try:
            file = open(self.state_dir, 'rb')
            if os.path.getsize(self.state_dir) > 0:
                self.prob_tree = pickle.load(file)
                self.prob_tree.validate()
                self.size = len(self.prob_tree.res)
        except FileNotFoundError:
            file = open(self.state_dir, 'wb+')
        file.close()


if __name__ == '__main__':
    mem = Memory()
    mem.memorize(Sample(State('str', 1, 1, 1), State('str2', 2, 2, 2), 3, 4, False))
