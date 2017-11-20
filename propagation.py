#!/usr/bin/env python3

from layers import *
from losses import *
from utils import classification_error


def forwardpropagation(x, model, args, training):
    K = len(model["W"])
    cache = {"g": [None] * K, "fc": [None] * K, "dp": [None] * (K - 1)}
    h = x

    for k in range(K):
        a, cache["fc"][k] = fully_connected(model["W"][k], model["b"][k], h)
        h, cache["g"][k] = model["g"][k](a)
        if k < K - 1 and args.use_dropout:
            h, cache["dp"][k] = dropout(h, training)

    return h, cache


def backpropagation(x, y, model, loss, dloss, args):
    K = len(model["W"])
    grad = {"dW": [None] * K, "db": [None] * K, "da": [None] * K}
    h, cache = forwardpropagation(x, model, args, True)

    l, y_pred = loss(h, y)
    da = dloss(y_pred, y)

    for k in range(K - 1, -1, -1):
        grad["dW"][k], grad["db"][k], dh = dfully_connected(da, cache["fc"][k])
        grad["da"][k] = da
        if k > 0:
            if args.use_dropout:
                dh = ddropout(dh, cache["dp"][k - 1])
            da = model["dg"][k - 1](dh, cache["g"][k - 1])

    return l, y_pred, grad
