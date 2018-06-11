import numpy as np
import os
import pickle
from dqn.utils.bean import STNode


class SumTree:
    L = 0
    R = 1

    def __init__(self, save_dir=None):
        self.res = []  # reservoir
        self.size = 0
        self.res.append(STNode(val=-1, sum=-1, data=None, idx=-1))
        self.save_dir = save_dir

    def _get_child(self, idx, lr):
        idx = idx * 2 + lr
        return idx if 0 <= idx <= self.size else -1

    def set(self, value, data=None, idx=None):
        if idx is None:
            self.res.append(STNode(val=value, sum=0, data=data, idx=self.size))
            self.size = len(self.res) - 1
            idx = - 1
            old_val = 0
        else:
            idx += 1  # skip dummy
            old_val = self.res[idx].val
            self.res[idx].val = value
        delta = value - old_val

        self._update_sum(idx, delta)
        self._shift(idx, delta)

    def rmv(self, idx):
        if idx == -1:
            idx = self.size - 1
        idx += 1
        if idx < 0 or idx >= self.size:
            return
        # move the last out
        fill = self.res[-1]
        self._update_sum(-1, -fill.val)
        self.res.pop()  # faster than del
        self.size = len(self.res) - 1
        # restore and move in
        fill.sum = self.res[idx].sum
        delta = fill.val - self.res[idx].val
        self.res[idx] = fill
        self._update_sum(idx, delta)
        self._shift(idx, delta)

    def get_item(self, value):
        idx = 1
        while idx >= 1:
            lc = self._get_child(idx, self.L)
            if lc != -1:
                test = value - self.res[lc].sum
                if test < 0:
                    idx = lc
                    continue
                value = test

            test = value - self.res[idx].val
            if test < 0:
                return self.res[idx], idx - 1
            value = test

            rc = self._get_child(idx, self.R)
            if rc != -1:
                test = value - self.res[rc].sum
                if test < 0:
                    idx = rc
                    continue
                return None, -1
            return None, -1

    def _update_sum(self, idx, delta):
        idx = idx if idx != -1 else self.size
        # along its (inclusive) path to root
        while idx > 0:
            self.res[idx].sum += delta
            idx //= 2

    # exchange with parent
    def _xchange_w_parent(self, c):
        p = c // 2
        self.res[p], self.res[c] = self.res[c], self.res[p]
        parent = self.res[p]
        child = self.res[c]
        # parent's sum remains the same (p_sum = p_val + l_sum + (r_val + (r_l_sum + r_r_sum)))
        parent.sum, child.sum = child.sum, parent.sum - parent.val + child.val
        parent.idx, child.idx = child.idx, parent.idx

    def _shift(self, idx, option):
        idx = idx if idx != -1 else self.size
        if option > 0:  # shift toward root
            while idx != 1:
                current = self.res[idx]
                parent = self.res[idx // 2]
                if parent.val < current.val:
                    self._xchange_w_parent(idx)
                    idx //= 2
                else:
                    break
        else:  # shift toward leaf
            while True:
                lc = self._get_child(idx, self.L)
                rc = self._get_child(idx, self.R)
                # choose the existing bigger one
                if lc == -1:
                    heir = rc
                elif rc == -1:
                    heir = lc
                else:
                    heir = lc if self.res[lc].val > self.res[rc].val else rc
                if heir == -1 or self.res[idx].val > self.res[heir].val:  # nothing to exchange
                    return
                else:
                    self._xchange_w_parent(heir)
                idx = heir

    def validate(self):
        valid = True
        for idx, node in enumerate(self.res):
            if idx == 0:
                continue
            l_c = self._get_child(idx, self.L)
            lcs = 0 if l_c == -1 else self.res[l_c].sum
            lcv = -1 if l_c == -1 else self.res[l_c].val
            r_c = self._get_child(idx, self.R)
            rcs = 0 if r_c == -1 else self.res[r_c].sum
            rcv = -1 if r_c == -1 else self.res[r_c].val
            if abs(self.res[idx].sum - (lcs + rcs + self.res[idx].val)) > 1e-3:
                print('invalid: sum {} != {} + {} + {}'.format(self.res[idx].sum, lcs, rcs, self.res[idx].val))
                valid = False
            if self.res[idx].val < lcv:
                print('invalid: value {} < {}(l child)'.format(self.res[idx].val, lcv))
                valid = False
            if self.res[idx].val < rcv:
                print('invalid: value {} < {}(r child)'.format(self.res[idx].val, rcv))
                valid = False
            if self.res[idx].idx + 1 != idx:
                print('invalid: {} != {}'.format(self.res[idx].idx + 1, idx))
                valid = False
        if valid:
            print('SumTree {} is valid'.format(self))

    def save(self, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir
        file = open(save_dir, 'wb')
        pickle.dump(self, file)
        file.close()

    def restore(self, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir
        try:
            file = open(save_dir, 'rb')
            if os.path.getsize(save_dir) > 0:
                cpy = pickle.load(file)
                cpy.validate()
                self.res = list(cpy.res)
                self.size = cpy.size
        except FileNotFoundError:
            file = open(save_dir, 'wb+')
        file.close()

    def __getitem__(self, idx):
        if idx == -1:
            return self.res[-1]
        return self.res[idx + 1]


def test1(st):
    # probability test
    hits = [0] * 100
    lengths = []
    for idx in range(0, 100):
        num = int(np.random.random() * 1000)
        lengths.append(num)
        st.set(value=num, data=idx)
    st.validate()
    for _ in range(0, 10000):
        num = np.random.random() * st.res[1].sum
        #_, index = st.get_item(num)
        j = st.get_item(num)[0].data
        # j = np.random.randint(0, len(hits))
        hits[j] += 1
    p = np.array(lengths)
    p = p / np.sum(p)
    p_ = np.array(hits)
    p_ = p_ / np.sum(p_)
    err = (p - p_) / p * p  # weighted acc to hit count
    for idx, e in enumerate(err):
        print(idx, e)
    print('err:', np.mean(np.abs(err)))


def test2(st):
    r_size = 1000  # tree size
    t_size = 40  # test size
    v_range = 1000  # val range
    for idx in range(0, r_size):
        num = np.random.randint(0, v_range)
        st.set(value=num, data=idx)
    data = {}  # indices to change
    new_val = np.random.randint(0, v_range, t_size)
    indices = np.random.randint(0, r_size, t_size)
    for idx in range(0, t_size):
        data[st.res[indices[idx]+1].data] = new_val[idx]
        st.set(value=new_val[idx], idx=indices[idx])
    st.validate()
    for idx in range(0, r_size):
        for d, v in data.items():
            if st.res[idx].data == d:
                print('found')
                if st.res[idx].val != v:
                    print('{}: {} != new_val({})'.format(idx, st.res[idx].val, v))
                    return
    print('pass')

def test3(st):
    r_size = 100  # tree size
    t_size = 4  # test size
    v_range = 1000  # val range
    for idx in range(0, r_size):
        num = np.random.randint(0, v_range)
        st.set(value=num, data=idx)
    indices = np.random.randint(0, r_size, t_size)
    for idx in range(0, t_size):
        st.rmv(indices[idx])
    st.validate()


if __name__ == '__main__':
    st = SumTree()
    test1(st)

    for idx in range(0, 100):
        num = np.inf
        st.set(value=num, data=idx)
    print(st.get_item(100))
