#!/usr/bin/env python3

import numpy as np


def l1(W):
    return np.sum(np.abs(W)), np.sign(W)


def l2(W):
    return np.sum(W * W), 2 * W
