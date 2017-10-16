# coding: utf-8

import functools
import hashlib
import math

import numpy as np
import pandas

import NeuralNetwork as nN

SUBSET_NUM = 5


def preprocess_train(data_raw) -> (np.ndarray, np.ndarray):
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


def preprocess_test(data_raw) -> np.ndarray:
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


def run(x: np.ndarray, y: np.ndarray, *, hidden_units: int, iter_num: int, friction: float, learning_rate: float) -> (
        float, float):
    train_cost, validation_cost = 0., 0.
    x = normalize(x)
    for cv_id in range(SUBSET_NUM):
        idx_train = np.ones(x.shape[1], np.bool)
        idx_train[np.arange(cv_id, y.shape[1], SUBSET_NUM)] = 0
        idx_validate = np.logical_not(idx_train)
        x_train, y_train = x[:, idx_train], y[:, idx_train]
        x_validate, y_validate = x[:, idx_validate], y[:, idx_validate]
        w, b = nN.init((x_train.shape[0], hidden_units, y_train.shape[0]), distribution="UNIFORM", dev_type="FAN_IN")
        w, b = nN.optimize(w, b, x_train, y_train, iter_num=iter_num, friction=friction, learning_rate=learning_rate)
        y_train_predicted = nN.forward_propagation(w, b, x_train)[-1]
        y_validate_predicted = nN.forward_propagation(w, b, x_validate)[-1]
        train_cost = train_cost + nN.cost(y_train, y_train_predicted)
        validation_cost = validation_cost + nN.cost(y_validate, y_validate_predicted)
        # print(np.sum(np.logical_xor(y_train, y_train_predicted >= 0.5)) / y_train.shape[1])
        # print(np.sum(np.logical_xor(y_validate, y_validate_predicted >= 0.5)) / y_validate.shape[1])
    return train_cost / SUBSET_NUM, validation_cost / SUBSET_NUM


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
    data_x, data_y = preprocess_train(pandas.read_csv("train.csv"))
    param_list = ((10, 1000, 0.1, 0.1),)
    param_list_hash = hashlib.sha256(str(param_list).encode('utf-8')).hexdigest()
    output_status = check_output_status("output.csv", param_list_hash)
    run_list = tuple(map(lambda params: functools.partial(run, x=data_x, y=data_y,
                                                          hidden_units=params[0],
                                                          iter_num=params[1],
                                                          friction=params[2],
                                                          learning_rate=params[3]),
                         param_list))
    if output_status == -1:
        restart("output.csv", param_list_hash, run_list)
    else:
        resume("output.csv", run_list, output_status)


main()

# def test_main():
#     data_train_subsets = preprocess_train(pandas.read_csv("train.csv"))
#     data_test_x = normalize(preprocess_test(pandas.read_csv("test.csv")))
#     run_list = ((0.000625, 12, 0.1, 0.08, 0.5, 0.0125),
#                 (0.02, 18, 0.1, 0.08, 0.5, 0.2),
#                 (0.02, 20, 0.1, 0.08, 0.5, 0.2))
#     neg_exp_cost = 0.0
#     result = []
#     for sigma, hidden_units, friction, learning_rate, dropout_rate, l2_decay in run_list:
#         cost, al = run_test(data_train_subsets, data_test_x, sigma, hidden_units, friction, learning_rate,
#                             dropout_rate, l2_decay)
#         print(al)
#         neg_exp_cost = neg_exp_cost + np.exp(-cost)
#         result.append((cost, al))
#     al_total = np.zeros((1, data_test_x.shape[1]))
#     for cost, al in result:
#         al_total = al_total + (np.exp(-cost) / neg_exp_cost) * al
#     print(al_total)
#     print(al_total >= 0.5)
#     # al1 = nn1.forward_propagation(data_test_x, dropout_rate=0.5)[-1]
#     # al2 = nn2.forward_propagation(data_test_x, dropout_rate=0.5)[-1]
#     # al3 = nn3.forward_propagation(data_test_x, dropout_rate=0.5)[-1]
#     # print(al1.shape, al2.shape, al3.shape)
#     # y = np.mean(np.concatenate((al1, al2, al3), axis=0), axis=0, keepdims=True) >= 0.5
#     # print(y)


# # main()
# # test_main()
# def test():
#     data_train_subsets = preprocess_train(pandas.read_csv("train.csv"))
#     for cv_id in range(len(data_train_subsets)):
#         data_train, data_validation = {}, {}
#         for i in range(len(data_train_subsets)):
#             if cv_id == i:
#                 data_validation = data_train_subsets[i]
#             else:
#                 if data_train:
#                     data_train["x"] = np.concatenate((data_train["x"], data_train_subsets[i]["x"]), axis=1)
#                     data_train["y"] = np.concatenate((data_train["y"], data_train_subsets[i]["y"]), axis=1)
#                 else:
#                     data_train = data_train_subsets[i]
#         data_train["x"] = normalize(data_train["x"])
#         data_validation["x"] = normalize(data_validation["x"])
#         w, b = nN.init((data_train["x"].shape[0], 10, data_train["y"].shape[0]),
#                        distribution="UNIFORM", dev_type="FAN_IN")
#         w, b = nN.optimize(w, b, data_train["x"], data_train["y"],
#                            iter_num=1000, friction=0.1, learning_rate=0.1)
#         y = nN.forward_propagation(w, b, data_validation["x"])[-1]
#         print(nN.cost(data_validation["y"], y))
#         print(np.sum(np.logical_xor(data_validation["y"], y >= 0.5)) / y.shape[1])
#
#
# test()
