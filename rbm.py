#!/usr/bin/env python3

import sys
import argparse
from utils import *
from activations import *
from losses import *
from optimizers import *


def h_activation(W, b, v):
    return sigmoid(np.dot(v, W) + b)[0]


def v_activation(W, c, h):
    return sigmoid(np.dot(h, W.transpose()) + c)[0]


def sample_h_given_v(W, b, v):
    return bernoulli(h_activation(W, b, v))


def sample_v_given_h(W, c, h):
    return bernoulli(v_activation(W, c, h))


def gibbs_sampling(K, W, b, c, v):
    v_sample = v

    for k in range(K):
        h_sample = sample_h_given_v(W, b, v_sample)
        v_sample = sample_v_given_h(W, c, h_sample)

    return v_sample


def calculate_gradients(K, W, b, c, v):
    T = v.shape[0]
    m, n = W.shape

    v_sample = gibbs_sampling(K, W, b, c, v)
    hv = h_activation(W, b, v)
    hv_sample = h_activation(W, b, v_sample)

    dW = (np.dot(v.transpose(), hv) - np.dot(v_sample.transpose(), hv_sample)) / T
    db = np.reshape(np.mean(hv - hv_sample, axis=0), (1, n))
    dc = np.reshape(np.mean(v - v_sample, axis=0), (1, m))

    return dW, db, dc


def contrastive_divergence(x_train, x_valid, W, b, c, optimizer, args):
    parameters = ["W", "b", "c"]
    model = {"W": [W], "b": [b], "c": [c]}
    T = x_train.shape[0]
    batches = ceil(T / args.batch_size)
    velocities = [{}] * batches
    for batch in range(batches):
        v = velocities[batch]
        for param in parameters:
            v["d" + param + "1"] = [np.zeros(mk.shape) for mk in model[param]]
            v["d" + param + "2"] = [np.zeros(mk.shape) for mk in model[param]]

    error_train = []
    error_valid = []
    for epoch in range(args.epochs):
        batch = 0
        epoch_error = 0

        for t in range(0, T, args.batch_size):
            t_end = T if t + args.batch_size > T else t + args.batch_size
            x_batch = x_train[range(t, t_end)]

            dW, db, dc = calculate_gradients(args.cd_k, W, b, c, x_batch)
            x_reconstruction = v_activation(W, c, h_activation(W, b, x_batch))
            error, _ = binary_cross_entropy(x_reconstruction, x_batch)

            epoch_error += error

            optimizer(parameters, model, {"dW": [-dW], "db": [-db], "dc": [-dc]},
                      velocities[batch], epoch + 1, args)
            batch = batch + 1

        x_reconstruction = v_activation(W, c, h_activation(W, b, x_valid))
        error, _ = binary_cross_entropy(x_reconstruction, x_valid)

        error_train.append(epoch_error / batches)
        error_valid.append(error)

        print("\nepoch %i" % epoch)
        print("train reconstruction error %f" % error_train[-1])
        print("validation reconstruction error %f" % error_valid[-1])

    return error_train, error_valid


def save_checkpoint(W, b, c, args):
    file = open(args.checkpoint, "w")
    m, n = W.shape

    file.write(str(m) + "\n")
    file.write(str(n) + "\n")

    N = m * n
    write_array(file, np.reshape(W, (1, N)), N)
    write_array(file, b, n)
    write_array(file, c, m)


def load_checkpoint(args):
    file = open(args.checkpoint, "r")

    m = int(file.readline())
    n = int(file.readline())

    W = np.fromstring(file.readline(), sep=',')
    W = np.reshape(W, (m, n))

    b = np.fromstring(file.readline(), sep=',')
    b = np.reshape(b, (1, n))

    c = np.fromstring(file.readline(), sep=',')
    c = np.reshape(c, (1, m))

    return W, b, c


def initialize_parameters(m, n):
    W = np.random.normal(0.0, 0.1, (m, n))
    b = np.zeros((1, n))
    c = np.zeros((1, m))

    return W, b, c


def train(args):
    x_train, y_train = load_data(args.file_train)
    x_train, _ = shuffle(x_train, y_train)

    x_valid, _ = load_data(args.file_valid)

    W, b, c = initialize_parameters(x_train.shape[1], 100)
    optimizer = sgd if args.optimizer == "sgd" else adam

    error_train, error_valid = contrastive_divergence(x_train, x_valid,
                                                      W, b, c, optimizer, args)

    plot_losses(error_train, error_valid, args.epochs,
                "reconstruction error", "reconstruction_error.png")
    plot_weights({"W": [W]}, 1)

    save_checkpoint(W, b, c, args)


def test(args):
    x_test = np.random.rand(100, 784)
    W, b, c = load_checkpoint(args)

    x_reconstruction = gibbs_sampling(1000, W, b, c, x_test)
    plot_reconstruction(x_reconstruction, 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-file_train", help="File with training data",
                        type=str, default="")
    parser.add_argument("-file_valid", help="File with validation data",
                        type=str, default="")
    parser.add_argument("-file_test", help="File with test data",
                        type=str, default="")
    parser.add_argument("-checkpoint", help="Checkpoint file",
                        type=str, default="checkpoint.txt")
    parser.add_argument("-epochs", help="Number of epochs",
                        type=int, default=300)
    parser.add_argument("-batch_size", help="Batch size",
                        type=int, default=30)
    parser.add_argument("-learning_rate", help="Learning rate",
                        type=float, default=0.1)
    parser.add_argument("-optimizer", help="Type of optimizer [sgd, adam]",
                        type=str, default="sgd")
    parser.add_argument("-momentum", help="Momentum",
                        type=float, default=0.0)
    parser.add_argument("-cd_k", help="The number of Contrastive Divergence steps",
                        type=int, default=1)

    args = parser.parse_args()

    if args.file_train != "" and args.file_valid != "":
        train(args)

    elif args.file_test != "":
        test(args)


if __name__ == '__main__':
    main()
