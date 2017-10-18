# coding: utf-8

import functools

import numpy as np

from neural_network.Dropout import Dropout
from neural_network.EarlyStop import EarlyStop
from neural_network.L2Decay import L2Decay


def init(layer_dims: tuple, *, distribution: str, dev_type: str, dropout: Dropout = None) -> (tuple, tuple):
    w, b = [], []
    layer_dims_p = dropout.init() if dropout is not None else layer_dims
    for i in range(1, len(layer_dims)):
        var = 0
        if dev_type == "FAN_IN":
            var = 1.0 / layer_dims_p[i - 1]
        elif dev_type == "FAN_IN_OUT":
            var = 2.0 / (layer_dims_p[i - 1] + layer_dims_p[i])
        if distribution == "NORMAL":
            w.append(np.random.randn(layer_dims[i], layer_dims[i - 1]) * np.sqrt(var))
        elif distribution == "UNIFORM":
            r = np.sqrt(3.0 * var)
            w.append(np.random.uniform(-r, r, (layer_dims[i], layer_dims[i - 1])))
        b.append(np.zeros((layer_dims[i], 1)))
    return tuple(w), tuple(b)


def forward_propagation(w: tuple, b: tuple, x: np.ndarray, *, training: bool, dropout: Dropout = None) -> tuple:
    a = [x]
    last = len(w)
    for l in range(1, last):
        al = np.dot(w[l - 1], a[l - 1]) + b[l - 1]
        np.maximum(al, 0, out=al)
        if dropout is not None:
            al = dropout.forward_propagation(al, layer=l, training=training)
        a.append(al)
    al = np.dot(w[last - 1], a[last - 1]) + b[last - 1]
    al = 1.0 / (1.0 + np.exp(-al))
    a.append(al)
    return tuple(a)


def back_propagation(w: tuple, y: np.ndarray, a: tuple, *, dropout: Dropout = None, l2_decay: L2Decay = None) -> (
        tuple, tuple):
    last = len(w)
    dz, dw, db = [None] * (last + 1), [None] * last, [None] * last
    dz[last] = (a[last] - y) / y.shape[1]
    for l in reversed(range(last)):
        dw[l] = np.dot(dz[l + 1], a[l].T)
        if dropout is not None:
            dw[l] = dropout.back_propagation(dw[l], layer=l)
        db[l] = np.sum(dz[l + 1], axis=1, keepdims=True)
        dz[l] = np.dot(w[l].T, dz[l + 1]) * (a[l] > 0)
    if l2_decay is not None:
        dw = l2_decay.back_propagation(tuple(dw), w)
    return tuple(dw), tuple(db)


def cost(y: np.ndarray, al: np.ndarray) -> np.ndarray:
    def stable_log(x: np.ndarray):
        return np.log(np.maximum(x, 1e-20))

    # return -np.mean(y * np.log(al) + (1 - y) * np.log(1.0 - al))
    return -np.mean(y * stable_log(al) + (1 - y) * stable_log(1.0 - al))


def gradient_check(w: tuple, b: tuple, x: np.ndarray, y: np.ndarray, *, eps: float = 1e-8):
    """Does not support L2 decay"""
    last = len(w)
    dw, db = [None] * last, [None] * last
    a = forward_propagation(w, b, x, training=True)
    tdw, tdb = back_propagation(w, y, a)
    for l in range(last):
        dw[l] = np.zeros(w[l].shape)
        delta = np.zeros(w[l].shape)
        for i in range(w[l].shape[0]):
            for j in range(w[l].shape[1]):
                delta[i, j] = eps
                c1 = cost(y, forward_propagation(w[l] + delta, b[l], x, training=False)[-1])
                c2 = cost(y, forward_propagation(w[l] - delta, b[l], x, training=False)[-1])
                delta[i, j] = 0.0
                dw[l][i, j] = (c1 - c2) / (eps * 2.0)
        db[l] = np.zeros(b[l].shape)
        delta = np.zeros(b[l].shape)
        for i in range(w[l].shape[0]):
            for j in range(b[l].shape[1]):
                delta[i, j] = eps
                c1 = cost(y, forward_propagation(w[l], b[l] + delta, x, training=False)[-1])
                c2 = cost(y, forward_propagation(w[l], b[l] - delta, x, training=False)[-1])
                delta[i, j] = 0.0
                dw[l][i, j] = (c1 - c2) / (eps * 2.0)
        print("w[%d]  std_err = %f ; b[%d]  std_err = %f" % (
            l, np.linalg.norm(tdw[l] - dw[l]), l, np.linalg.norm(tdb[l] - db[l])))


def gradient_descent_momentum_update(w0: tuple, b0: tuple, vw0: tuple, vb0: tuple, dw: tuple, db: tuple, *,
                                     friction: float, learning_rate: float) -> (tuple, tuple, tuple, tuple):
    w, b, vw, vb = [], [], [], []
    for l in range(len(w0)):
        vw.append((1.0 - friction) * vw0[l] + dw[l])
        vb.append((1.0 - friction) * vb0[l] + db[l])
        w.append(w0[l] - learning_rate * vw[l])
        b.append(b0[l] - learning_rate * vb[l])
    return tuple(w), tuple(b), tuple(vw), tuple(vb)


def optimize(w: tuple, b: tuple, x: np.ndarray, y: np.ndarray, *,
             iter_num: int, friction: float, learning_rate: float,
             dropout: Dropout = None, early_stop: EarlyStop = None, l2_decay: L2Decay = None) -> (tuple, tuple):
    vw, vb = tuple(np.zeros(wl.shape) for wl in w), tuple(np.zeros(bl.shape) for bl in b)
    for i in range(iter_num):
        if dropout is not None:
            dropout = dropout.sample()
        dw, db = back_propagation(w, y, forward_propagation(w, b, x, training=True, dropout=dropout),
                                  dropout=dropout, l2_decay=l2_decay)
        w, b, vw, vb = gradient_descent_momentum_update(w, b, vw, vb, dw, db, friction=friction,
                                                        learning_rate=learning_rate)
        if early_stop is not None:
            learning_rate = early_stop.new_learning_rate(learning_rate, i, w, b,
                                                         functools.partial(forward_propagation,
                                                                           training=False, dropout=dropout), cost)
            if learning_rate is None:
                break
    if early_stop is not None:
        return early_stop.best_w, early_stop.best_b
    return w, b
