# coding: utf-8

import hashlib
import math
import random

import numpy as np
import pandas

import NeuralNetwork as nn

SUBSET_NUM = 5


def preprocess_train(data_raw) -> list:
    data = []
    urn = []
    for person in data_raw.get_values():
        urn.append(person)
    random.shuffle(urn)
    subset_size = (len(urn) + SUBSET_NUM - 1) // SUBSET_NUM
    cnt = 0
    for person in urn:
        if cnt % subset_size == 0:
            data.append({"x": [], "y": []})
        cnt = cnt + 1
        data[-1]["y"].append(person[1])
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
        data[-1]["x"].append(vec)
    for j in range(SUBSET_NUM):
        data[j]["x"] = np.array(data[j]["x"]).T
        data[j]["y"] = np.array(data[j]["y"]).reshape((1, -1))
    return data


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


def run(data_subsets, sigma, hidden_units, friction, learning_rate, dropout_rate, l2_decay):
    train_cost, validation_cost = 0., 0.
    for ti in range(1):
        for cv_id in range(len(data_subsets)):
            data_train, data_validation = {}, {}
            for i in range(len(data_subsets)):
                if cv_id == i:
                    data_validation = data_subsets[i]
                else:
                    if data_train:
                        data_train["x"] = np.concatenate((data_train["x"], data_subsets[i]["x"]), axis=1)
                        data_train["y"] = np.concatenate((data_train["y"], data_subsets[i]["y"]), axis=1)
                    else:
                        data_train = data_subsets[i]
            data_train["x"] = normalize(data_train["x"])
            # nn = NeuralNetwork((data_train["x"].shape[0], hidden_units, data_train["y"].shape[0]), sigma=sigma)
            train_cost_cur, validation_cost_cur = nn.optimize(data_train["x"],
                                                              data_train["y"],
                                                              data_validation["x"],
                                                              data_validation["y"],
                                                              dropout_rate=dropout_rate,
                                                              l2_decay=l2_decay,
                                                              optimization_params={
                                                                  "f": friction,
                                                                  "learning_rate": learning_rate})
            train_cost = train_cost + train_cost_cur
            validation_cost = validation_cost + validation_cost_cur
    return train_cost, validation_cost


def run_test(data_subsets, data_test_x, sigma, hidden_units, friction, learning_rate, dropout_rate, l2_decay):
    cv_id = random.randint(0, len(data_subsets) - 1)
    data_train, data_validation = {}, {}
    for i in range(len(data_subsets)):
        if cv_id == i:
            data_validation = data_subsets[i]
        else:
            if data_train:
                data_train["x"] = np.concatenate((data_train["x"], data_subsets[i]["x"]), axis=1)
                data_train["y"] = np.concatenate((data_train["y"], data_subsets[i]["y"]), axis=1)
            else:
                data_train = data_subsets[i]
    # mean = np.mean(np.concatenate((data_train["x"], data_test_x), axis=1), axis=1, keepdims=True)
    # std = np.std(np.concatenate((data_train["x"], data_test_x), axis=1), axis=1, keepdims=True)
    mean = np.mean(data_train["x"], axis=1, keepdims=True)
    std = np.std(data_train["x"], axis=1, keepdims=True)
    data_train["x"] = normalize(data_train["x"], mean, std)
    # nn = NeuralNetwork((data_train["x"].shape[0], hidden_units, data_train["y"].shape[0]), sigma=sigma)
    train_cost, validation_cost = nn.optimize(data_train["x"], data_train["y"],
                                              data_validation["x"], data_validation["y"],
                                              dropout_rate=dropout_rate,
                                              l2_decay=l2_decay,
                                              optimization_params={
                                                  "f": friction,
                                                  "learning_rate": learning_rate})
    print(train_cost, validation_cost)
    return validation_cost, nn.forward_propagation(normalize(data_test_x, mean, std), dropout_rate=dropout_rate)[-1]


def check_output_status(file_name: str, run_list_hash: str) -> int:
    try:
        with open(file_name, 'r') as f:
            if f.readline().split(',')[0] == run_list_hash:
                cnt = 0
                while len(f.readline().split(',')) >= 2:
                    cnt = cnt + 1
                return cnt
            else:
                return -1
    except IOError:
        return -1


def restart(data_train_subsets: list, file_name: str, run_list_hash: str, run_list: tuple):
    with open(file_name, 'w') as f:
        f.write(run_list_hash + ',\n')
        f.flush()
        for sigma, hidden_units, friction, learning_rate, dropout_rate, l2_decay in run_list:
            train_cost, validation_cost = run(data_train_subsets, sigma, hidden_units, friction, learning_rate,
                                              dropout_rate, l2_decay)
            f.write(str(train_cost) + ',' + str(validation_cost) + ',\n')
            f.flush()


def resume(data_train_subsets: list, file_name: str, run_list: tuple, start_pos: int):
    run_list = run_list[start_pos:]
    with open(file_name, 'a') as f:
        for sigma, hidden_units, friction, learning_rate, dropout_rate, l2_decay in run_list:
            train_cost, validation_cost = run(data_train_subsets, sigma, hidden_units, friction, learning_rate,
                                              dropout_rate, l2_decay)
            f.write(str(train_cost) + ',' + str(validation_cost) + ',\n')
            f.flush()


def main():
    data_train_subsets = preprocess_train(pandas.read_csv("train.csv"))
    # run_list = tuple((sigma, hidden_units, 0.1, learning_rate, 0.5, l2_decay)
    #                  for sigma in (0.02, 0.04, 0.08)
    #                  for hidden_units in (18, 20)
    #                  for learning_rate in (0.08,)
    #                  for l2_decay in (0.2, 0.4, 0.8))
    run_list = ((0.000625 * 2, 12, 0.1, 0.08, 0.5, 0.025 * 2),)
    run_list_hash = hashlib.sha256(str(run_list).encode('utf-8')).hexdigest()
    output_status = check_output_status("output.csv", run_list_hash)
    if output_status == -1:
        restart(data_train_subsets, "output.csv", run_list_hash, run_list)
    else:
        resume(data_train_subsets, "output.csv", run_list, output_status)


def test_main():
    data_train_subsets = preprocess_train(pandas.read_csv("train.csv"))
    data_test_x = normalize(preprocess_test(pandas.read_csv("test.csv")))
    run_list = ((0.000625, 12, 0.1, 0.08, 0.5, 0.0125),
                (0.02, 18, 0.1, 0.08, 0.5, 0.2),
                (0.02, 20, 0.1, 0.08, 0.5, 0.2))
    neg_exp_cost = 0.0
    result = []
    for sigma, hidden_units, friction, learning_rate, dropout_rate, l2_decay in run_list:
        cost, al = run_test(data_train_subsets, data_test_x, sigma, hidden_units, friction, learning_rate,
                            dropout_rate, l2_decay)
        print(al)
        neg_exp_cost = neg_exp_cost + np.exp(-cost)
        result.append((cost, al))
    al_total = np.zeros((1, data_test_x.shape[1]))
    for cost, al in result:
        al_total = al_total + (np.exp(-cost) / neg_exp_cost) * al
    print(al_total)
    print(al_total >= 0.5)
    # al1 = nn1.forward_propagation(data_test_x, dropout_rate=0.5)[-1]
    # al2 = nn2.forward_propagation(data_test_x, dropout_rate=0.5)[-1]
    # al3 = nn3.forward_propagation(data_test_x, dropout_rate=0.5)[-1]
    # print(al1.shape, al2.shape, al3.shape)
    # y = np.mean(np.concatenate((al1, al2, al3), axis=0), axis=0, keepdims=True) >= 0.5
    # print(y)


# main()
# test_main()
def test():
    data_train_subsets = preprocess_train(pandas.read_csv("train.csv"))
    for cv_id in range(len(data_train_subsets)):
        data_train, data_validation = {}, {}
        for i in range(len(data_train_subsets)):
            if cv_id == i:
                data_validation = data_train_subsets[i]
            else:
                if data_train:
                    data_train["x"] = np.concatenate((data_train["x"], data_train_subsets[i]["x"]), axis=1)
                    data_train["y"] = np.concatenate((data_train["y"], data_train_subsets[i]["y"]), axis=1)
                else:
                    data_train = data_train_subsets[i]
        data_train["x"] = normalize(data_train["x"])
        data_validation["x"] = normalize(data_validation["x"])
        w, b = nn.init((data_train["x"].shape[0], 10, data_train["y"].shape[0]),
                       distribution="UNIFORM", dev_type="FAN_IN")
        w, b = nn.optimize(w, b, data_train["x"], data_train["y"],
                           iter_num=1000, friction=0.1, learning_rate=0.1)
        y = nn.forward_propagation(w, b, data_validation["x"])[-1]
        print(nn.cost(data_validation["y"], y))
        print(np.sum(np.logical_xor(data_validation["y"], y >= 0.5)) / y.shape[1])


test()
