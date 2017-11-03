#!/usr/bin/env python3

import sys
import argparse
from utils import *
from activations import *
from losses import *
from regularizers import *
from propagation import *
from optimizers import *


def optimize(x_train, x_valid, model, regularizer, optimizer, args):
    K = len(model["W"])
    parameters = ["W", "b"]
    T = x_train.shape[0]
    batches = ceil(T / args.batch_size)
    velocities = [{}] * batches
    for batch in range(batches):
        v = velocities[batch]
        for param in parameters:
            v["d" + param + "1"] = [np.zeros(mk.shape) for mk in model[param]]
            v["d" + param + "2"] = [np.zeros(mk.shape) for mk in model[param]]

    loss_train = []
    loss_valid = []

    y_train = np.copy(x_train)
    if args.use_dae:
        for t in range(T):
            index = np.where(x_train[t] > 0)[0]
            m = int(index.shape[0] * 0.25)
            index = index[np.random.choice(index.shape[0], m, replace=False)]
            x_train[t][index] = 0

    for epoch in range(args.epochs):
        batch = 0
        epoch_total_loss = 0
        epoch_loss = 0

        for t in range(0, T, args.batch_size):
            t_end = T if t + args.batch_size > T else t + args.batch_size
            loss, _, grad = backpropagation(x_train[range(t, t_end)],
                                            y_train[range(t, t_end)],
                                            model, binary_cross_entropy,
                                            dbinary_cross_entropy, args)
            omega = 0
            for k in range(K):
                reg, dreg = regularizer(model["W"][k])
                omega += reg
                grad["dW"][k] += args.regularizer_strength * dreg

            epoch_total_loss += loss + args.regularizer_strength * omega
            epoch_loss += loss

            optimizer(parameters, model, grad,
                      velocities[batch], epoch + 1, args)
            batch = batch + 1

        h, _ = forwardpropagation(x_valid, model, args, False)
        loss, _ = binary_cross_entropy(h, x_valid)

        loss_train.append(epoch_loss / batches)
        loss_valid.append(loss)

        print("\nepoch %i" % epoch)
        print("average train loss %f" % (epoch_total_loss / batches))
        print("train cross entropy loss %f" % loss_train[-1])
        print("validation cross entropy loss %f" % loss_valid[-1])

    return loss_train, loss_valid


def initialize_parameters(layer_sizes):
    model = {"W": [], "b": []}
    K = len(layer_sizes) - 1
    k = 0

    for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):
        model["W"].append(np.random.normal(0.0, 0.1, (m, n)))
        model["b"].append(np.zeros((1, n)))
        k = k + 1

    return model


def train(args):
    x_train, y_train = load_data(args.file_train)
    x_train, _ = shuffle(x_train, y_train)

    x_valid, _ = load_data(args.file_valid)

    layer_sizes = [x_train.shape[1], 100, x_train.shape[1]]
    model = initialize_parameters(layer_sizes)
    model["g"] = [sigmoid] * (len(layer_sizes) - 1)
    model["dg"] = [dsigmoid] * (len(layer_sizes) - 1)
    regularizer = l2 if args.regularizer == "l2" else l1
    optimizer = sgd if args.optimizer == "sgd" else adam

    loss_train, loss_valid = optimize(x_train, x_valid, model,
                                      regularizer, optimizer, args)

    plot_losses(loss_train, loss_valid, args.epochs,
                "cross entropy loss", "cross_entropy.png")
    plot_weights(model, 1)

    model["g"] = ["sigmoid"] * (len(layer_sizes) - 1)
    save_checkpoint(args, model)


def test(args):
    x_test, _ = load_data(args.file_test)
    model = load_checkpoint(args)

    h, _ = forwardpropagation(x_test, model, args, False)
    loss, x_reconstruction = binary_cross_entropy(h, x_test)

    print("test cross entropy loss %f" % loss)
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
    parser.add_argument("-regularizer", help="Type of regularizer [l1, l2]",
                        type=str, default="l2")
    parser.add_argument("-regularizer_strength", help="Regularizer strength",
                        type=float, default=0.0)
    parser.add_argument("-optimizer", help="Type of optimizer [sgd, adam]",
                        type=str, default="sgd")
    parser.add_argument("-momentum", help="Momentum",
                        type=float, default=0.0)
    parser.add_argument("-use_dropout", help="Use dropout",
                        action="store_true")
    parser.add_argument("-use_dae", help="Use denoising autoencoder",
                        action="store_true")

    args = parser.parse_args()

    if args.file_train != "" and args.file_valid != "":
        train(args)

    elif args.file_test != "":
        test(args)


if __name__ == '__main__':
    main()
