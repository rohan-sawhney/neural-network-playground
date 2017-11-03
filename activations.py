#!/usr/bin/env python3

import numpy as np


def linear(a):
    h = a
    cache = a

    return h, cache


def dlinear(dh, h):
    return dh * np.ones(h.shape)


def sigmoid(a):
    h = 1 / (1 + np.exp(-a))
    cache = h

    return h, cache


def dsigmoid(dh, h):
    return dh * h * (1 - h)


def tanh(a):
    h = np.tanh(a)
    cache = h

    return h, cache


def dtanh(dh, h):
    return dh * (1 - h * h)


def relu(a):
    h = np.maximum(0, a)
    cache = a

    return h, cache


def drelu(dh, a):
    return dh * (a >= 0).astype(np.float64)


def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)
