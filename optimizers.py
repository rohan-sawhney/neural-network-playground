#!/usr/bin/env python3

import numpy as np


def sgd(parameters, model, grad, velocity, epoch, args):
    alpha = args.learning_rate
    beta = args.momentum

    for param in parameters:
        m = model[param]
        g = grad["d" + param]
        v = velocity["d" + param + "1"]

        for k in range(len(m)):
            v[k] = alpha * g[k] + beta * v[k]
            m[k] -= v[k]


def adam(parameters, model, grad, velocity, epoch, args):
    alpha = args.learning_rate
    beta1 = 0.9
    beta2 = 0.999

    for param in parameters:
        m = model[param]
        g = grad["d" + param]
        v1 = velocity["d" + param + "1"]
        v2 = velocity["d" + param + "2"]

        for k in range(len(m)):
            v1[k] = (1 - beta1) * g[k] + beta1 * v1[k]
            v2[k] = (1 - beta2) * g[k]**2 + beta2 * v2[k]

            v1_hat = v1[k] / (1 - beta1**epoch)
            v2_hat = v2[k] / (1 - beta2**epoch)

            m[k] -= alpha * v1_hat / (np.sqrt(v2_hat) + 1e-8)
