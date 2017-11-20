#!/usr/bin/env python3

import re
import collections
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import floor, ceil, sqrt
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


def load_corpus(filename):
    file = open(filename, "r")
    corpus = []

    for line in file:
        line = line.strip()  # remove \n \r
        if line:
            line = re.sub(' +', ' ', line)  # remove consecutive spaces
            corpus.append(line)

    return corpus


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


def save_checkpoint_language(args, model, word_2_idx):
    file = open(args.checkpoint, "w")

    V = args.vocab_size
    D = args.embedding_size
    file.write(str(V) + "\n")
    file.write(str(D) + "\n")

    N = V * D
    write_array(file, np.reshape(model["C"][0], (1, N)), N)

    for word, idx in word_2_idx.items():
        file.write(word + " " + str(idx) + "\n")

    K = len(model["W"])
    file.write(str(K) + "\n")
    for k in range(K):
        m, n = model["W"][k].shape
        file.write(str(m) + "\n")
        file.write(str(n) + "\n")

        N = m * n
        write_array(file, np.reshape(model["W"][k], (1, N)), N)
        write_array(file, model["b"][k], n)

        file.write(model["g"][k] + "\n")


def load_checkpoint_language(args):
    file = open(args.checkpoint, "r")
    model = {"W": [], "b": [], "g": [], "C": []}

    V = int(file.readline())
    D = int(file.readline())
    C = np.fromstring(file.readline(), sep=',')
    model["C"].append(np.reshape(C, (V, D)))

    word_2_idx = {}
    idx_2_word = {}
    for i in range(V):
        line = file.readline()
        word, idx = line.split()
        idx = int(idx)
        word_2_idx[word] = idx
        idx_2_word[idx] = word

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

    return model, word_2_idx, idx_2_word


def plot_losses(loss_train, loss_valid, epochs, ylabel, name):
    plt.clf()

    x = range(epochs)
    if len(loss_train) > 0:
        plt.plot(x, loss_train, label="train")
    if len(loss_valid) > 0:
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


def plot_ngram_distribution(x, y, idx_2_word, display_count):
    counter = collections.Counter()
    for i in range(len(x)):
        counter.update({tuple(x[i] + [y[i]]): 1})

    ngrams = []
    frequency = []
    most_common = counter.most_common(len(x))
    print("%i most common sequences: " % display_count)
    for i in range(len(most_common)):
        ngram = most_common[i]
        ngrams.append(i)
        frequency.append(ngram[1])

        if i < display_count:
            ngram_str = ""
            for j in range(len(ngram[0])):
                ngram_str += idx_2_word[ngram[0][j]]
                ngram_str += "\t"
            print(ngram_str)

    plt.clf()
    plt.plot(ngrams, frequency)
    plt.ylabel("Frequency")
    plt.xlabel("Unique ngrams")
    plt.savefig("ngram_distribution.png")


def plot_embeddings(C, idx, idx_2_word):
    C_idx = C[idx]
    x = C_idx[:, 0].tolist()
    y = C_idx[:, 1].tolist()

    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i in range(len(idx)):
        ax.annotate(idx_2_word[idx[i]], (x[i], y[i]))

    plt.savefig("embedding_2D.png")
