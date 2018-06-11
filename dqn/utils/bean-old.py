import numpy as np
import copy


class State:
    def __init__(self,
                 img: np.ndarray or None,
                 time: int or None,
                 step: int or None,
                 time_tag: int or None):
        self.img = img
        self.time = time
        self.step = step
        self.time_tag = time_tag

    def __str__(self):
        return str(self.__dict__)


class Sample:
    def __init__(self,
                 p_state: State or None,
                 state: State or None,
                 action: int or None,
                 reward: int or None,
                 terminal: bool or None):
        self.p_state = p_state
        self.state = state
        self.action = action
        self.reward = reward
        self.terminal = terminal

    def __str__(self):
        return str(self.__dict__)

    def __copy__(self):
        cpy = object.__new__(self.__class__)
        cpy.__dict__.update(self.__dict__)
        cpy.state, cpy.p_state = copy.copy(self.state), copy.copy(self.p_state)
        return cpy


class STNode:
    def __init__(self, val: float, sum: float, data, idx: int):
        self.val = val
        self.sum = sum
        self.idx = idx
        self.data = data

    def __copy__(self):
        cpy = object.__new__(self.__class__)
        cpy.__dict__.update(self.__dict__)
        cpy.data = copy.copy(self.data)

    def __str__(self):
        return str(self.__dict__)
