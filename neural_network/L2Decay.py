# coding: utf-8

import numpy as np


class L2Decay:
    def __init__(self, *, factor: float):
        self.factor = factor

    def back_propagation(self, dw: tuple, w: tuple) -> tuple:
        return tuple([dw[i] + w[i] * self.factor for i in range(len(dw))])

    @staticmethod
    def weight_variance(w: tuple) -> float:
        return sum([np.sum(w[i] ** 2) for i in range(len(w))])
