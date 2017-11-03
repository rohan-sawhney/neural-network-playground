#!/usr/bin/env python3

import numpy as np


def fully_connected(W, b, h):
    a = np.dot(h, W) + b
    cache = (W, h)

    return a, cache


def dfully_connected(da, cache):
    W, h = cache

    dW = np.dot(h.T, da)
    db = np.sum(da, axis=0, keepdims=True)
    dh = np.dot(da, W.T)

    return dW, db, dh


def dropout(h, training):
    mask = np.random.binomial(1, 0.5, size=h.shape) if training else 0.5
    cache = mask

    return h * mask, cache


def ddropout(dh, mask):
    return dh * mask
