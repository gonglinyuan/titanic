# coding: utf-8

import numpy as np


class NeuralNetwork:
    def __init__(self, layer_dims: tuple, sigma: float = 0.01):
        self.w, self.b, self.layer_dims = [], [], layer_dims
        for i in range(1, len(layer_dims)):
            self.w.append(np.random.randn(layer_dims[i], layer_dims[i - 1]) * sigma)
            self.b.append(np.zeros((layer_dims[i], 1)))

    def forward_propagation(self, x: np.ndarray, dropout_mask: list = None, dropout_rate: float = None,
                            training: bool = False) -> list:
        a = [x]
        nl = len(self.layer_dims)
        for l in range(1, nl - 1):
            al = np.dot(self.w[l - 1], a[l - 1]) + self.b[l - 1]
            np.maximum(al, 0, out=al)
            if not (dropout_rate is None):
                if training:
                    al = al * dropout_mask[l]
                else:
                    al = al * (1. - dropout_rate)
            a.append(al)
        al = np.dot(self.w[nl - 2], a[nl - 2]) + self.b[nl - 2]
        np.clip(al, -30., 30., al)
        al = 1. / (1. + np.exp(-al))
        a.append(al)
        return a

    def back_propagation(self, y: np.ndarray, a: list, dropout_mask: list = None) -> (list, list):
        nl = len(self.layer_dims)
        dz, dw, db = [None] * nl, [None] * (nl - 1), [None] * (nl - 1)
        dz[nl - 1] = (a[nl - 1] - y) / y.shape[1]
        for l in reversed(range(nl - 1)):
            dw[l] = np.dot(dz[l + 1], a[l].T)
            if not (dropout_mask is None):
                dw[l] = dw[l] * dropout_mask[l + 1]
            db[l] = np.sum(dz[l + 1], axis=1, keepdims=True)
            dz[l] = np.dot(self.w[l].T, dz[l + 1]) * (a[l] > 0)
        return dw, db

    @staticmethod
    def cost(y: np.ndarray, al: np.ndarray) -> np.ndarray:
        return -np.mean(y * np.log(al) + (1 - y) * np.log(1. - al))

    def gradient_check(self, x: np.ndarray, y: np.ndarray, eps: float = 1e-8):
        nl = len(self.layer_dims)
        dw, db = [None] * (nl - 1), [None] * (nl - 1)
        a = self.forward_propagation(x)
        tdw, tdb = self.back_propagation(y, a)
        for l in range(nl - 1):
            dw[l] = np.zeros(self.w[l].shape)
            for i in range(self.w[l].shape[0]):
                for j in range(self.w[l].shape[1]):
                    self.w[l][i, j] = self.w[l][i, j] + eps
                    a = self.forward_propagation(x)
                    c1 = self.cost(y, a[len(a) - 1])
                    self.w[l][i, j] = self.w[l][i, j] - eps * 2.
                    a = self.forward_propagation(x)
                    c2 = self.cost(y, a[len(a) - 1])
                    self.w[l][i, j] = self.w[l][i, j] + eps
                    dw[l][i, j] = (c1 - c2) / (eps * 2.)
            db[l] = np.zeros(self.b[l].shape)
            for i in range(self.b[l].shape[0]):
                for j in range(self.b[l].shape[1]):
                    self.b[l][i, j] = self.b[l][i, j] + eps
                    a = self.forward_propagation(x)
                    c1 = self.cost(y, a[len(a) - 1])
                    self.b[l][i, j] = self.b[l][i, j] - eps * 2.
                    a = self.forward_propagation(x)
                    c2 = self.cost(y, a[len(a) - 1])
                    self.b[l][i, j] = self.b[l][i, j] + eps
                    db[l][i, j] = (c1 - c2) / (eps * 2.)
            print(np.linalg.norm(tdw[l] - dw[l]), np.linalg.norm(tdb[l] - db[l]))
            print(np.linalg.norm((tdw[l] - dw[l]) / dw[l]), np.linalg.norm((tdb[l] - db[l]) / db[l]))

    def gradient_descent_update(self, dw: list, db: list, params=None) -> dict:
        if params is None:
            params = {"learning_rate": 0.7}
        for l in range(len(self.layer_dims) - 1):
            self.w[l] = self.w[l] - params["learning_rate"] * dw[l]
            self.b[l] = self.b[l] - params["learning_rate"] * db[l]
        return {}

    def gradient_descent_momentum_update(self, dw: list, db: list, cache: dict, params=None) -> dict:
        if params is None:
            params = {"f": 0.1, "learning_rate": 0.02}
        if not cache:
            cache = {"v_w": [], "v_b": []}
            for l in range(len(self.layer_dims) - 1):
                cache["v_w"].append(np.zeros(self.w[l].shape))
                cache["v_b"].append(np.zeros(self.b[l].shape))
        for l in range(len(self.layer_dims) - 1):
            cache["v_w"][l] = (1. - params["f"]) * cache["v_w"][l] + dw[l]
            cache["v_b"][l] = (1. - params["f"]) * cache["v_b"][l] + db[l]
            self.w[l] = self.w[l] - params["learning_rate"] * cache["v_w"][l]
            self.b[l] = self.b[l] - params["learning_rate"] * cache["v_b"][l]
        return cache

    def optimize(self, x: np.ndarray, y: np.ndarray, x_cv: np.ndarray, y_cv: np.ndarray,
                 optimization_params: dict = None, iter_num: int = 1500, dropout_rate: float = None,
                 l2_decay: float = 0.) -> (float, float):
        best_so_far = {"cost": np.infty, "w": None, "b": None, "iter_num": 0}
        cache = {}
        no_update_cnt = 0
        for i in range(iter_num):
            if dropout_rate is None:
                a = self.forward_propagation(x, training=True)
                dw, db = self.back_propagation(y, a)
            else:
                dropout_mask = [np.ones((x.shape[0], 1))]
                for dim in self.layer_dims[1:-1]:
                    dropout_mask.append(np.random.rand(dim, 1) >= dropout_rate)
                dropout_mask.append(np.ones((self.layer_dims[-1], 1)))
                a = self.forward_propagation(x, dropout_mask=dropout_mask, dropout_rate=dropout_rate, training=True)
                dw, db = self.back_propagation(y, a, dropout_mask=dropout_mask)
            for l in range(len(self.layer_dims) - 1):
                dw[l] = dw[l] + self.w[l] * l2_decay
                db[l] = db[l] + self.b[l] * l2_decay
            cache = self.gradient_descent_momentum_update(dw, db, cache, optimization_params)
            a = self.forward_propagation(x_cv, dropout_rate=dropout_rate)
            cost = self.cost(y_cv, a[-1])
            if cost < best_so_far["cost"]:
                best_so_far["cost"] = cost
                best_so_far["w"] = self.w
                best_so_far["b"] = self.b
                best_so_far["iter_num"] = i + 1
                no_update_cnt = 0
            else:
                no_update_cnt = no_update_cnt + 1
            if no_update_cnt % 10 == 0:
                optimization_params["learning_rate"] = optimization_params["learning_rate"] * 0.5
            if no_update_cnt >= 30:
                break
        self.w = best_so_far["w"]
        self.b = best_so_far["b"]
        # print(best_so_far["iter_num"])
        return self.cost(y, self.forward_propagation(x, dropout_rate=dropout_rate)[-1]), self.cost(
            y_cv, self.forward_propagation(x_cv, dropout_rate=dropout_rate)[-1])

    def predict(self, x: np.ndarray, dropout_rate: float = None):
        a = self.forward_propagation(x, dropout_rate=dropout_rate)
        return a[len(self.layer_dims) - 1] >= 0.5
