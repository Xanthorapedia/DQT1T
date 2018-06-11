import numpy as np
import copy
from dqn.utils.sum_tree import SumTree
from dqn.utils.LRUCache import LRUCache
from dqn.utils.bean import Sample
import os


class Memory:
    IMG_DIR = './train_mem'  # all (preprocessed) screenshots
    HST_PKL = IMG_DIR + './history.pkl'  # everything else except images

    def __init__(self):
        self.img_max_cap = 1000
        self.img_cache = LRUCache(self.img_max_cap, save_dir=self.IMG_DIR)
        self.max_prob = 20
        self.prob_tree = SumTree(self.HST_PKL)
        self.prob_tree.restore()
        # self.mem = []
        self.size = 0
        # storing screenshots
        if not os.path.isdir(self.IMG_DIR):
            os.mkdir(self.IMG_DIR)
        self.load_mem()

    def memorize(self, history: Sample):
        if self.size == self.img_max_cap:
            self.prob_tree.rmv(-1)
        self.register(history)
        history_cpy = copy.copy(history)
        history_cpy.state.img = history_cpy.p_state.img = None  # to save space
        # self.mem.append(history_tmp)
        # update prob_tree
        max_prob = self.max_prob if self.prob_tree.size == 0 else self.prob_tree[0].val
        self.prob_tree.set(value=max_prob, data=history_cpy)
        self.size += 1
        # backup
        if self.size % 4 == 0:
            self.save_mem()

    def register(self, game_state: Sample):
        state = game_state.state
        p_state = game_state.p_state
        if game_state.terminal:
            state.time_tag = '0'
        else:
            self.img_cache.set(state.time_tag, state.img)
        self.img_cache.set(p_state.time_tag, p_state.img)

    def update_prob(self, losses: list, items):
        losses = np.array(losses)
        prob = abs(losses) + 1.0 / self.prob_tree.size  # at least random
        # shrink max prob every time to avoid stucking on a fixed set of samples
        less_than_max = [val for val in prob if val < self.max_prob]
        # if less_than_max:
        #     self.max_prob = max(less_than_max)# + 0.9 * self.max_prob
        for loss, itm in zip(prob, items):
            self.prob_tree.set(min(loss, self.max_prob), idx=itm.idx)

    def sample(self, batch_size):
        samples = []
        refs = []
        val = np.random.rand(batch_size) * self.prob_tree[0].sum
        for v in val:
            item, ref = self.get_mem(v)
            while item is None:
                v = np.random.rand() * self.prob_tree[0].sum
                item, ref = self.get_mem(v)
            samples.append(item)
            refs.append(ref)
        return samples, refs

    @staticmethod
    def unpack(samples, refs):
        states, p_states = [], []
        actions, rewards, terminals = [], [], []
        for s in samples:
            states.append(s.state.img)
            p_states.append(s.p_state.img)
            # []: dim0: batch
            actions.append([s.action])
            rewards.append([s.reward])
            terminals.append([s.terminal])
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
        if item is None:
            return None, None
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
        return self.img_cache.get(time_tag)

    def save_mem(self, save_cache=True, save_tree=True):
        if save_tree:
            self.prob_tree.save()
        if save_cache:
            self.img_cache.save()

    def load_mem(self):
        self.prob_tree.restore()
        self.size = self.prob_tree.size


if __name__ == '__main__':
    HST_PKL = '../train_mem/history.pkl'
    pt0 = SumTree(save_dir=HST_PKL)
    pt0.restore()
    pt = SumTree(save_dir=HST_PKL)
    for idx, item in enumerate(pt0.res):
        if item.idx >= 0:
            # print(idx, item)
            pt.set(value=10, data=item.data)
    pt.set(value=0, idx=0)
    pt.validate()
    for idx, (i0, i1) in enumerate(zip(pt0.res, pt.res)):
        if id(i0.data) != id(i1.data):
            print(idx, id(i0.data), id(i1.data))
    pt.save()
