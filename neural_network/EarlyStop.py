# coding: utf-8

from typing import *

import numpy as np


class EarlyStop:
    def __init__(self, x: np.ndarray, y: np.ndarray, *, interval: int, half_life: int, threshold: float):
        assert half_life % interval == 0
        self.x, self.y = x, y
        self.interval, self.half_life, self.threshold = interval, half_life, threshold
        self.best_cost, self.best_epoch, self.best_w, self.best_b, self.counter = None, None, tuple(), tuple(), 0

    def new_learning_rate(self, learning_rate: float, epoch: int, w: tuple, b: tuple,
                          forward_propagation: Callable, cost: Callable) -> Union[float, type(None)]:
        if epoch % self.interval == 0:
            cur_cost = cost(self.y, forward_propagation(w, b, self.x)[-1])
            # print(cur_cost)
            if self.best_cost is None or cur_cost <= self.best_cost:
                self.best_cost, self.best_epoch, self.best_w, self.best_b, self.counter = cur_cost, epoch, w, b, 0
            else:
                self.counter = self.counter + 1
                if self.counter * self.interval >= self.half_life:
                    self.counter = 0
                    learning_rate = learning_rate / 2.0
                    if learning_rate < self.threshold:
                        # print("epoch = ", epoch)
                        return None
        return learning_rate
