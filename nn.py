#!/usr/bin/env python3

import sys
import argparse
from utils import *
from activations import *
from losses import *
from regularizers import *
from propagation import *
from optimizers import *


def optimize(x_train, y_train, x_valid, y_valid, model, regularizer, optimizer, args):
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
    error_train = []
    loss_valid = []
    error_valid = []

    for epoch in range(args.epochs):
        batch = 0
        epoch_total_loss = 0
        epoch_loss = 0
        epoch_error = 0

        for t in range(0, T, args.batch_size):
            t_end = T if t + args.batch_size > T else t + args.batch_size
            loss, prob, grad = backpropagation(x_train[range(t, t_end)],
                                               y_train[range(t, t_end)],
                                               model, cross_entropy,
                                               dcross_entropy, args)
            omega = 0
            for k in range(K):
                reg, dreg = regularizer(model["W"][k])
                omega += reg
                grad["dW"][k] += args.regularizer_strength * dreg

            epoch_total_loss += loss + args.regularizer_strength * omega
            epoch_loss += loss
            epoch_error += classification_error(prob, y_train[range(t, t_end)])

            optimizer(parameters, model, grad,
                      velocities[batch], epoch + 1, args)
            batch = batch + 1

        h, _ = forwardpropagation(x_valid, model, args, False)
        loss, prob = cross_entropy(h, y_valid)
        error = classification_error(prob, y_valid)

        loss_train.append(epoch_loss / batches)
        loss_valid.append(loss)
        error_train.append(epoch_error / batches)
        error_valid.append(error)

        print("\nepoch %i" % epoch)
        print("average train loss %f" % (epoch_total_loss / batches))
        print("train cross entropy loss %f" % loss_train[-1])
        print("validation cross entropy loss %f" % loss_valid[-1])
        print("train classification error %.2f%%" % error_train[-1])
        print("validation classification error %.2f%%" % error_valid[-1])

    return loss_train, loss_valid, error_train, error_valid


def initialize_parameters(layer_sizes, args):
    model = {"W": [], "b": []}
    K = len(layer_sizes) - 1
    k = 0

    for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):
        if k == 0 and args.use_checkpoint_weights:
            checkpoint = load_checkpoint(args)
            model["W"].append(checkpoint["W"][0])

        else:
            u = np.sqrt(6 / (m + n))
            model["W"].append(np.random.uniform(-u, u, (m, n)))

        model["b"].append(np.zeros((1, n)))
        k = k + 1

    return model


def train(args):
    x_train, y_train = load_data(args.file_train)
    x_train, y_train = shuffle(x_train, y_train)

    x_valid, y_valid = load_data(args.file_valid)

    layer_sizes = [x_train.shape[1], 100, args.num_classes]
    model = initialize_parameters(layer_sizes, args)
    model["g"] = [sigmoid] * (len(layer_sizes) - 2) + [linear]
    model["dg"] = [dsigmoid] * (len(layer_sizes) - 2) + [dlinear]
    regularizer = l2 if args.regularizer == "l2" else l1
    optimizer = sgd if args.optimizer == "sgd" else adam

    loss_train, loss_valid, error_train, error_valid = optimize(x_train, y_train,
                                                                x_valid, y_valid, model,
                                                                regularizer, optimizer, args)

    plot_losses(loss_train, loss_valid, args.epochs,
                "cross entropy loss", "cross_entropy.png")
    plot_losses(error_train, error_valid, args.epochs,
                "mean classification error %", "classification_error.png")
    plot_weights(model, 1)

    model["g"] = ["sigmoid"] * (len(layer_sizes) - 2) + ["linear"]
    save_checkpoint(args, model)


def test(args):
    x_test, y_test = load_data(args.file_test)
    model = load_checkpoint(args)

    h, _ = forwardpropagation(x_test, model, args, False)
    loss, prob = cross_entropy(h, y_test)
    error = classification_error(prob, y_test)

    print("test cross entropy loss %f" % loss)
    print("test classification error %.2f%%" % error)


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
    parser.add_argument("-num_classes", help="Number of classes",
                        type=int, default=10)
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
    parser.add_argument("-use_checkpoint_weights", help="Use checkpoint weights for training",
                        action="store_true")
    args = parser.parse_args()

    if args.file_train != "" and args.file_valid != "":
        train(args)

    elif args.file_test != "":
        test(args)


if __name__ == '__main__':
    main()
