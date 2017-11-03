#!/usr/bin/env python3

import numpy as np
from activations import softmax


def binary_cross_entropy(y_pred, y):
    T = y.shape[0]
    loss = -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) / T

    return loss, y_pred


def dbinary_cross_entropy(y_pred, y):
    T = y.shape[0]
    da = (y_pred - y) / T

    return da


def cross_entropy(y_pred, y):
    T = y.shape[0]
    prob = softmax(y_pred)
    loss = np.sum(-np.log(prob[range(T), y])) / T

    return loss, prob


def dcross_entropy(prob, y):
    T = y.shape[0]
    da = np.copy(prob)
    da[range(T), y] -= 1
    da /= T

    return da
