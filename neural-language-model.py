#!/usr/bin/env python3

import sys
import argparse
import collections
from utils import *
from activations import *
from losses import *
from regularizers import *
from propagation import *
from optimizers import *


def tokenize_corpus(corpus):
    words = []
    for line in corpus:
        words.append("START")

        tokens = line.split()
        for token in tokens:
            words.append(token.lower())

        words.append("END")

    return words


def generate_ngrams(words, n):
    ngrams = zip(*[words[i:] for i in range(n)])
    split = list(zip(*ngrams))
    X = list(zip(*split[:len(split) - 1]))
    X = [list(x) for x in X]
    Y = list(split[-1])

    return X, Y


def generate_vocab(words, vocab_size):
    # build truncated vocab from most frequent words
    counter = collections.Counter()
    for word in words:
        counter.update({word: 1})

    vocab = counter.most_common(vocab_size - 1)
    vocab.append(("UNK", 1))

    # create word to id and id to word dictionaries
    idx = range(vocab_size)
    vocab = [v[0] for v in vocab]
    word_2_idx = dict(zip(vocab, idx))
    idx_2_word = dict(zip(idx, vocab))

    return word_2_idx, idx_2_word


def convert_words_to_idx(words, word_2_idx):
    idx = []
    for word in words:
        if word in word_2_idx:
            idx.append(word_2_idx[word])
        else:
            idx.append(word_2_idx["UNK"])

    return idx


def initialize_parameters(layer_sizes):
    model = {"W": [], "b": []}
    K = len(layer_sizes) - 1
    k = 0

    for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):
        model["W"].append(np.random.normal(0.0, 0.1, (m, n)))
        model["b"].append(np.zeros((1, n)))
        k = k + 1

    return model


def build_embedding_matrix(x, C):
    B = len(x)
    N = len(x[0])
    D = C.shape[1]
    x_embedding = np.zeros((B, N * D))
    for i in range(B):
        off = 0
        for j in range(N):
            x_embedding[i, range(off, off + D)] = C[x[i][j]]
            off += D

    return x_embedding


def compute_embedding_grad(x, C, W, da):
    B = len(x)
    N = len(x[0])
    D = C.shape[1]
    dC = np.zeros(C.shape)
    dh = np.dot(da, W.T)
    for i in range(B):
        off = 0
        for j in range(N):
            dC[x[i][j]] += dh[i, range(off, off + D)]
            off += D

    return dC


def optimize(x_train, y_train, x_valid, y_valid, model, regularizer, optimizer, args):
    K = len(model["W"])
    T = len(x_train)
    C = model["C"][0]
    parameters = ["W", "b", "C"]
    batches = ceil(T / args.batch_size)
    velocities = [{}] * batches
    for batch in range(batches):
        v = velocities[batch]
        for param in parameters:
            v["d" + param + "1"] = [np.zeros(mk.shape) for mk in model[param]]
            v["d" + param + "2"] = [np.zeros(mk.shape) for mk in model[param]]

    loss_train = []
    loss_valid = []
    perplexity_valid = []

    for epoch in range(args.epochs):
        batch = 0
        epoch_total_loss = 0
        epoch_loss = 0

        for t in range(0, T, args.batch_size):
            print("batch: ", t)
            t_end = T if t + args.batch_size > T else t + args.batch_size
            x_batch = x_train[t:t_end]
            y_batch = y_train[t:t_end]

            x_embedding = build_embedding_matrix(x_batch, C)
            y_embedding = np.array(y_batch)
            loss, prob, grad = backpropagation(x_embedding, y_embedding,
                                               model, cross_entropy,
                                               dcross_entropy, args)

            W = model["W"][0]
            da = grad["da"][0]
            grad["dC"] = [compute_embedding_grad(x_batch, C, W, da)]

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

        x_embedding = build_embedding_matrix(x_valid, C)
        y_embedding = np.array(y_valid)
        h, _ = forwardpropagation(x_embedding, model, args, False)
        loss, _ = cross_entropy(h, y_embedding)

        loss_train.append(epoch_loss / batches)
        loss_valid.append(loss)
        perplexity_valid.append(2**loss)

        print("\nepoch %i" % epoch)
        print("average train loss %f" % (epoch_total_loss / batches))
        print("train cross entropy loss %f" % loss_train[-1])
        print("validation cross entropy loss %f" % loss_valid[-1])
        print("validation perplexity %f" % perplexity_valid[-1])

    return loss_train, loss_valid, perplexity_valid


def train(args):
    # load training and validation words from corpus
    words_train = tokenize_corpus(load_corpus(args.file_train))
    words_valid = tokenize_corpus(load_corpus(args.file_valid))

    # generate vocabulary
    word_2_idx, idx_2_word = generate_vocab(words_train, args.vocab_size)

    # convert training and validation words to ids
    words_train_idx = convert_words_to_idx(words_train, word_2_idx)
    words_valid_idx = convert_words_to_idx(words_valid, word_2_idx)

    # generate training and validation data
    x_train, y_train = generate_ngrams(words_train_idx, args.n)
    x_valid, y_valid = generate_ngrams(words_valid_idx, args.n)

    plot_ngram_distribution(x_train, y_train, idx_2_word, 50)

    V = args.vocab_size
    D = args.embedding_size
    N = args.n

    layer_sizes = [(N - 1) * D, 128, V]
    model = initialize_parameters(layer_sizes)
    model["C"] = [np.random.normal(0.0, 0.1, (V, D))]
    model["g"] = [tanh] * (len(layer_sizes) - 2) + [linear]
    model["dg"] = [dtanh] * (len(layer_sizes) - 2) + [dlinear]
    regularizer = l2 if args.regularizer == "l2" else l1
    optimizer = sgd if args.optimizer == "sgd" else adam

    loss_train, loss_valid, perplexity_valid = optimize(x_train, y_train,
                                                        x_valid, y_valid, model,
                                                        regularizer, optimizer, args)

    plot_losses(loss_train, loss_valid, args.epochs,
                "cross entropy loss", "cross_entropy.png")
    plot_losses([], perplexity_valid, args.epochs,
                "perplexity", "perplexity.png")

    model["g"] = ["tanh"] * (len(layer_sizes) - 2) + ["linear"]
    save_checkpoint_language(args, model, word_2_idx)


def test(args):
    model, word_2_idx, idx_2_word = load_checkpoint_language(args)
    C = model["C"][0]

    if args.seed_sequence != "":
        sequence = args.seed_sequence
        words_test = sequence.split() + ["UNK"]
        words_test_idx = convert_words_to_idx(words_test, word_2_idx)
        x_test, _ = generate_ngrams(words_test_idx, args.n)

        for i in range(args.seed_sequence_length):
            x_embedding = build_embedding_matrix(x_test, C)
            h, _ = forwardpropagation(x_embedding, model, args, False)
            prob = softmax(h)
            prediction = idx_2_word[np.argmax(prob)]

            sequence += " " + prediction
            x_test = [x_test[0][1:] + [x_test[0][0]]]
            x_test[0][-1] = word_2_idx[prediction]

        print(sequence)

    if args.similar_words != "":
        similar_words = args.similar_words.split()
        idx = convert_words_to_idx(similar_words, word_2_idx)

        print(np.linalg.norm(C[idx[0]] - C[idx[1]]))

    if args.embedding_size == 2:
        idx = np.random.randint(0, args.vocab_size, 500)
        plot_embeddings(C, idx, idx_2_word)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-file_train", help="File with training data",
                        type=str, default="")
    parser.add_argument("-file_valid", help="File with validation data",
                        type=str, default="")
    parser.add_argument("-seed_sequence", help="Initial N - 1 word sequence for generation",
                        type=str, default="")
    parser.add_argument("-similar_words", help="Computes similarity between 2 words separated by space",
                        type=str, default="")
    parser.add_argument("-checkpoint", help="Checkpoint file",
                        type=str, default="checkpoint.txt")
    parser.add_argument("-vocab_size", help="Vocabulary size",
                        type=int, default=8000)
    parser.add_argument("-embedding_size", help="Embedding size",
                        type=int, default=16)
    parser.add_argument("-seed_sequence_length", help="Number of words to be generated",
                        type=int, default=10)
    parser.add_argument("-n", help="N-gram",
                        type=int, default=4)
    parser.add_argument("-epochs", help="Number of epochs",
                        type=int, default=100)
    parser.add_argument("-batch_size", help="Batch size",
                        type=int, default=1000)
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
    args = parser.parse_args()

    if args.file_train != "" and args.file_valid != "":
        train(args)

    elif args.seed_sequence != "" or args.similar_words != "":
        test(args)


if __name__ == '__main__':
    main()
