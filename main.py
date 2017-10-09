# coding: utf-8

import math
import random

import numpy as np
import pandas

from NeuralNetwork import NeuralNetwork

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


def normalize(x: np.ndarray) -> np.ndarray:
    t = x - np.mean(x, axis=1, keepdims=True)
    return t / np.std(t, axis=1, keepdims=True)


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
            nn = NeuralNetwork((data_train["x"].shape[0], hidden_units, data_train["y"].shape[0]), sigma=sigma)
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


# print(np.sum(np.abs(nn.predict(data_train["x"]) - data_train["y"])) / data_train["y"].shape[1])
# print(np.sum(np.abs(nn.predict(data_validation["x"]) - data_validation["y"])) / data_validation["y"].shape[1])


data_train_subsets = preprocess_train(pandas.read_csv("train.csv"))
# for sigma in (0.000625, 0.0025, 0.01, 0.04, 0.16):
#     for hidden_units in (8, 12, 16, 20):
#         for learning_rate in (0.08, ):
#             for l2_decay in (0.025, 0.1, 0.4):
#                 train_cost, validation_cost = run(data_train_subsets, sigma, hidden_units, 0.1, learning_rate, 0.5, l2_decay)
#                 print(train_cost, validation_cost)

# sigma != 0.16

for sigma in (0.000625 / 2, 0.000625, 0.000625 * 2):
    for hidden_units in (6, 8, 10, 12):
        for learning_rate in (0.08, ):
            for l2_decay in (0.025 / 2, 0.025, 0.025 * 2):
                train_cost, validation_cost = run(data_train_subsets, sigma, hidden_units, 0.1, learning_rate, 0.5, l2_decay)
                print(str(train_cost) + ',' + str(validation_cost) + ',')

for sigma in (0.02, 0.04, 0.08):
    for hidden_units in (18, 20):
        for learning_rate in (0.08, ):
            for l2_decay in (0.2, 0.4, 0.8):
                train_cost, validation_cost = run(data_train_subsets, sigma, hidden_units, 0.1, learning_rate, 0.5, l2_decay)
                print(str(train_cost) + ',' + str(validation_cost) + ',')