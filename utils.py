#!/usr/bin/env python3

from math import floor, ceil, sqrt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from activations import *


def shuffle(x, y):
    T = x.shape[0]
    p = np.random.permutation(T)

    return x[p], y[p]


def classification_error(prob, y):
    return np.mean(np.argmax(prob, axis=1) != y) * 100


def bernoulli(p):
    return (np.random.rand(*p.shape) < p).astype(int)


##########################################################################
# loading, saving, plotting

def load_data(filename):
    xy = np.loadtxt(filename, delimiter=",")
    T, D = xy.shape
    x = xy[range(T), 0:D - 1]
    y = xy[range(T), D - 1].astype(int)

    return x, y


def write_array(file, arr, N):
    for i in range(N):
        file.write(str(arr[0, i]) + ("\n" if i == N - 1 else ","))


def save_checkpoint(args, model):
    K = len(model["W"])
    file = open(args.checkpoint, "w")

    file.write(str(K) + "\n")
    for k in range(K):
        m, n = model["W"][k].shape
        file.write(str(m) + "\n")
        file.write(str(n) + "\n")

        N = m * n
        write_array(file, np.reshape(model["W"][k], (1, N)), N)
        write_array(file, model["b"][k], n)

        file.write(model["g"][k] + "\n")


def load_checkpoint(args):
    file = open(args.checkpoint, "r")
    model = {"W": [], "b": [], "g": []}

    K = int(file.readline())
    for k in range(K):
        m = int(file.readline())
        n = int(file.readline())

        Wk = np.fromstring(file.readline(), sep=',')
        model["W"].append(np.reshape(Wk, (m, n)))

        bk = np.fromstring(file.readline(), sep=',')
        model["b"].append(np.reshape(bk, (1, n)))

        gk = file.readline()
        if "linear" in gk:
            model["g"].append(linear)
        elif "sigmoid" in gk:
            model["g"].append(sigmoid)
        elif "tanh" in gk:
            model["g"].append(tanh)
        elif "relu" in gk:
            model["g"].append(relu)

    return model


def plot_losses(loss_train, loss_valid, epochs, ylabel, name):
    plt.clf()

    x = range(epochs)
    plt.plot(x, loss_train, label="train")
    plt.plot(x, loss_valid, label="validation")

    plt.ylabel(ylabel)
    plt.xlabel("epochs")
    plt.legend()

    plt.savefig(name)


def plot_weights(model, K):
    for k in range(K):
        m, n = model["W"][k].shape
        m = int(sqrt(m))
        Wk_reshaped = np.reshape(model["W"][k], (m, m, n))

        n_sqrt = sqrt(n)
        r = ceil(n_sqrt)
        c = floor(n_sqrt)

        plt.clf()
        plt.figure(figsize=(r, c))
        gs = gridspec.GridSpec(r, c)
        gs.update(wspace=0.0, hspace=0.0)

        for i in range(n):
            plt.subplot(gs[i])
            plt.axis("off")
            plt.imshow(Wk_reshaped[:, :, i], cmap="gray")

        plt.suptitle("Layer %i Learned Weights" % (k + 1))
        plt.savefig("W%i.png" % (k + 1))


def plot_reconstruction(x, K):
    xK = x[np.random.choice(x.shape[0], K, replace=False), :]
    K_sqrt = sqrt(K)
    r = ceil(K_sqrt)
    c = floor(K_sqrt)

    plt.clf()
    plt.figure(figsize=(r, c))
    gs = gridspec.GridSpec(r, c)
    gs.update(wspace=0.0, hspace=0.0)

    for k in range(K):
        m = xK[k].shape[0]
        m = int(sqrt(m))
        xk_reshaped = np.reshape(xK[k], (m, m))

        plt.subplot(gs[k])
        plt.axis("off")
        plt.imshow(xk_reshaped, cmap="gray")

    plt.suptitle("Reconstruction")
    plt.savefig("reconstruction.png")
