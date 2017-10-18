# coding: utf-8

import functools
import hashlib
import math

import numpy as np
import pandas

import neural_network.neural_network as nn
from neural_network.Dropout import Dropout
from neural_network.EarlyStop import EarlyStop
from neural_network.L2Decay import L2Decay

SUBSET_NUM = 5


def preprocess_train(data_raw: pandas.DataFrame) -> (np.ndarray, np.ndarray):
    x, y, urn = [], [], []
    for person in data_raw.get_values():
        urn.append(person)
    np.random.shuffle(urn)
    for person in urn:
        y.append(person[1])
        vec = [1 if person[2] == 1 else 0, 1 if person[2] == 2 else 0, 1 if person[4] == 'male' else 0]
        if math.isnan(person[5]):
            vec.append(0)
            vec.append(0)
        else:
            vec.append(person[5])
            vec.append(1)
        vec.append(person[6])
        vec.append(person[7])
        vec.append(person[9])
        vec.append(1 if person[11] == 'C' else 0)
        vec.append(1 if person[11] == 'Q' else 0)
        x.append(vec)
    return np.array(x).T, np.array(y).reshape((1, -1))


def preprocess_test(data_raw: pandas.DataFrame) -> np.ndarray:
    data = []
    for person in data_raw.get_values():
        vec = [1 if person[1] == 1 else 0, 1 if person[1] == 2 else 0, 1 if person[3] == 'male' else 0]
        if math.isnan(person[4]):
            vec.append(0)
            vec.append(0)
        else:
            vec.append(person[4])
            vec.append(1)
        vec.append(person[5])
        vec.append(person[6])
        vec.append(person[8])
        vec.append(1 if person[10] == 'C' else 0)
        vec.append(1 if person[10] == 'Q' else 0)
        data.append(vec)
    return np.array(data).T


def normalize(x: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None) -> np.ndarray:
    if mean is None:
        mean = np.mean(x, axis=1, keepdims=True)
    if std is None:
        std = np.std(x, axis=1, keepdims=True)
    return (x - mean) / std


def run(x: np.ndarray, y: np.ndarray, *, distribution: str, dev_type: str, hidden_units: int, iter_num: int,
        friction: float, learning_rate: float, dropout_rate: float = None, early_stop_config: dict = None,
        l2_decay_factor: float = None) -> (
        float, float):
    train_cost, validation_cost, train_acc, validation_acc = 0.0, 0.0, 0.0, 0.0
    x = normalize(x)
    for cv_id in range(SUBSET_NUM):
        idx_train = np.ones(x.shape[1], np.bool)
        idx_train[np.arange(cv_id, y.shape[1], SUBSET_NUM)] = 0
        idx_validate = np.logical_not(idx_train)
        x_train, y_train = x[:, idx_train], y[:, idx_train]
        x_validate, y_validate = x[:, idx_validate], y[:, idx_validate]
        layer_dims = (x_train.shape[0], hidden_units, y_train.shape[0])
        if dropout_rate is not None:
            dropout = Dropout(rate=dropout_rate, layer_dims=layer_dims)
        else:
            dropout = None
        if early_stop_config is not None:
            early_stop = EarlyStop(x_validate, y_validate,
                                   interval=early_stop_config["interval"], half_life=early_stop_config["half_life"],
                                   threshold=early_stop_config["threshold"])
        else:
            early_stop = None
        if l2_decay_factor is not None:
            l2_decay = L2Decay(factor=l2_decay_factor)
        else:
            l2_decay = None
        w, b = nn.init(layer_dims, distribution=distribution, dev_type=dev_type, dropout=dropout)
        w, b = nn.optimize(w, b, x_train, y_train,
                           iter_num=iter_num, friction=friction, learning_rate=learning_rate,
                           dropout=dropout, early_stop=early_stop, l2_decay=l2_decay)
        y_train_p = nn.forward_propagation(w, b, x_train, training=False, dropout=dropout)[-1]
        y_validate_p = nn.forward_propagation(w, b, x_validate, training=False, dropout=dropout)[-1]
        train_cost = train_cost + nn.cost(y_train, y_train_p)
        validation_cost = validation_cost + nn.cost(y_validate, y_validate_p)
        train_acc = train_acc + np.sum(np.logical_xor(y_train, y_train_p >= 0.5)) / y_train.shape[1]
        validation_acc = validation_acc + np.sum(np.logical_xor(y_validate, y_validate_p >= 0.5)) / y_validate.shape[1]
    train_acc, validation_acc = 1.0 - train_acc / SUBSET_NUM, 1.0 - validation_acc / SUBSET_NUM
    print(train_acc, validation_acc)
    return train_cost / SUBSET_NUM, validation_cost / SUBSET_NUM


def run_test(x_train: np.ndarray, y_train: np.ndarray, x_test, *,
             distribution: str, dev_type: str, hidden_units: int, iter_num: int, friction: float,
             learning_rate: float, dropout_rate: float = None, early_stop_config: dict = None,
             l2_decay_factor: float = None) -> np.ndarray:
    mean = np.mean(np.concatenate((x_train, x_test), axis=1), axis=1, keepdims=True)
    std = np.std(np.concatenate((x_train, x_test), axis=1), axis=1, keepdims=True)
    x_train, x_test = normalize(x_train, mean, std), normalize(x_test, mean, std)
    layer_dims = (x_train.shape[0], hidden_units, y_train.shape[0])
    if dropout_rate is not None:
        dropout = Dropout(rate=dropout_rate, layer_dims=layer_dims)
    else:
        dropout = None
    if early_stop_config is not None:
        early_stop = EarlyStop(x_train, y_train,
                               interval=early_stop_config["interval"], half_life=early_stop_config["half_life"],
                               threshold=early_stop_config["threshold"])
    else:
        early_stop = None
    if l2_decay_factor is not None:
        l2_decay = L2Decay(factor=l2_decay_factor)
    else:
        l2_decay = None
    w, b = nn.init(layer_dims, distribution=distribution, dev_type=dev_type, dropout=dropout)
    w, b = nn.optimize(w, b, x_train, y_train,
                       iter_num=iter_num, friction=friction, learning_rate=learning_rate, dropout=dropout,
                       early_stop=early_stop, l2_decay=l2_decay)
    y_test_p = nn.forward_propagation(w, b, x_test, training=False, dropout=dropout)[-1]
    return y_test_p >= 0.5


def check_output_status(file_name: str, param_list_hash: str) -> int:
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            if f.readline().split(',')[0] == param_list_hash:
                cnt = 0
                while len(f.readline().split(',')) >= 2:
                    cnt = cnt + 1
                return cnt
            else:
                return -1
    except IOError:
        return -1


def restart(file_name: str, param_list_hash: str, run_list: tuple) -> None:
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(param_list_hash + ',\n')
        f.flush()
        for fun in run_list:
            train_cost, validation_cost = fun()
            f.write(str(train_cost) + ',' + str(validation_cost) + ',\n')
            f.flush()


def resume(file_name: str, run_list: tuple, start_pos: int) -> None:
    run_list = run_list[start_pos:]
    with open(file_name, "a", encoding="utf-8") as f:
        for fun in run_list[start_pos:]:
            train_cost, validation_cost = fun()
            f.write(str(train_cost) + ',' + str(validation_cost) + ',\n')
            f.flush()


def main():
    x_train, y_train = preprocess_train(pandas.read_csv("train.csv"))
    # param_list = (("UNIFORM", "FAN_IN", 20, 1000, 0.1, 0.1, None, (5, 15, 0.005)),)
    param_list = tuple([("UNIFORM", "FAN_IN", hidden_units, 1000, 0.1, 0.7, None, None, l2_decay)
                        for hidden_units in (8, 12, 16, 20)
                        for l2_decay in (0.001, 0.01, 0.1)])
    # print(param_list)
    param_list_hash = hashlib.sha256(str(param_list).encode('utf-8')).hexdigest()
    output_status = check_output_status("output.csv", param_list_hash)
    run_list = tuple(map(lambda params: functools.partial(run, x=x_train, y=y_train,
                                                          distribution=params[0],
                                                          dev_type=params[1],
                                                          hidden_units=params[2],
                                                          iter_num=params[3],
                                                          friction=params[4],
                                                          learning_rate=params[5],
                                                          dropout_rate=params[6],
                                                          early_stop_config=None if params[7] is None else
                                                          {"interval": params[7][0],
                                                           "half_life": params[7][1],
                                                           "threshold": params[7][2]},
                                                          l2_decay_factor=params[8]),
                         param_list))
    if output_status == -1:
        restart("output.csv", param_list_hash, run_list)
    else:
        resume("output.csv", run_list, output_status)


def test():
    x_train, y_train = preprocess_train(pandas.read_csv("train.csv"))
    x_test = preprocess_test(pandas.read_csv("test.csv"))
    y_test = run_test(x_train, y_train, x_test, distribution="UNIFORM", dev_type="FAN_IN", hidden_units=20,
                      iter_num=1000, friction=0.1, learning_rate=0.1, dropout_rate=None,
                      early_stop_config={"interval": 5, "half_life": 15, "threshold": 0.005})
    pandas.DataFrame(y_test.astype(int).T,
                     index=np.arange(y_train.shape[1] + 1, y_train.shape[1] + y_test.shape[1] + 1),
                     columns=("Survived",)).to_csv("submission.csv", index_label="PassengerId")


main()
# test()
