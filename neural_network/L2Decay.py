# coding: utf-8

import numpy as np


class L2Decay:
    def __init__(self, *, factor: float):
        self.factor = factor

    def back_propagation(self, dw: tuple, w: tuple, m: int) -> tuple:
        return tuple([dw[i] + w[i] * self.factor / m for i in range(len(dw))])

    @staticmethod
    def weight_variance(w: tuple, m: int) -> float:
        return sum([np.sum(w[i] ** 2) for i in range(len(w))]) / (m * 2)
