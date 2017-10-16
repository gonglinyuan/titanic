# coding: utf-8

import numpy as np


class Dropout:
    def __init__(self, *, rate: float, layer_dims: tuple, mask: tuple = None):
        self.rate = rate
        self.layer_dims = layer_dims
        self.mask = mask

    def init(self) -> tuple:
        return tuple([self.layer_dims[i] * (self.rate if i > 1 and i + 1 < len(self.layer_dims) else 1.0)
                      for i in range(len(self.layer_dims))])

    def sample(self) -> object:
        return Dropout(rate=self.rate, layer_dims=self.layer_dims,
                       mask=tuple([np.ones((self.layer_dims[0], 1))] +
                                  [np.random.rand(dim, 1) >= self.rate for dim in self.layer_dims[1:-1]] +
                                  [np.ones((self.layer_dims[-1], 1))]))

    def forward_propagation(self, al: np.ndarray, *, layer: int, training: bool) -> np.ndarray:
        return al * (self.mask[layer] if training else (1.0 - self.rate))

    def back_propagation(self, dw: np.ndarray, *, layer: int) -> np.ndarray:
        return dw * self.mask[layer + 1]
