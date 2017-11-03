# neural-network-playground

This codebase contains an implementation of a feedforward neural network, a RBM, an autoencoder and a denoising autoencoder. To train these models on the MNIST dataset, type the any of the following commands on the command line:

```
python nn.py -file_train=digitstrain.txt -file_valid=digitsvalid.txt
python rbm.py -file_train=digitstrain.txt -file_valid=digitsvalid.txt
python autoencoder.py -file_train=digitstrain.txt -file_valid=digitsvalid.txt
python autoencoder.py.py -file_train=digitstrain.txt -file_valid=digitsvalid.txt -use_dae
```

By default, all models contain a single hidden layer with 100 sigmoid units and will train with a learning rate of 0.1 for 300 epochs. The relevant metadata such as cross entropy loss, classification error, reconstruction error are printed to the command line and plots of the learned weights are generated after training. A checkpoint file is also saved containing the model parameters which is required for inference.

To run inference with the above networks, type the any of the following commands on the command line:

```
python nn.py -file_test=digitstest.txt
python rbm.py -file_test=digitstest.txt
python autoencoder.py -file_test=digitstest.txt
```

The RBM model does not actually use the test data, it instead initializes 100 Gibbs chains  with random configurations of the visible variables, and runs the Gibbs sampler for 1000 steps. The weights of the first layer of the feedforward neural network can be initialized with weights of a RBM or autoencoder as follows:

```
python nn.py -file_test=digitstest.txt -use_checkpoint_weights -checkpoint:PATH_RBM_OR_AUTOENCODER_CHECKPOINT_FILE
```

Detailed command line options are listed below for all the models:

```
nn.py [-h] [-file_train FILE_TRAIN] [-file_valid FILE_VALID]
             [-file_test FILE_TEST] [-checkpoint CHECKPOINT]
             [-num_classes NUM_CLASSES] [-epochs EPOCHS]
             [-batch_size BATCH_SIZE] [-learning_rate LEARNING_RATE]
             [-regularizer REGULARIZER]
             [-regularizer_strength REGULARIZER_STRENGTH]
             [-optimizer OPTIMIZER] [-momentum MOMENTUM] [-use_dropout]
             [-use_checkpoint_weights]

rbm.py [-h] [-file_train FILE_TRAIN] [-file_valid FILE_VALID]
             [-file_test FILE_TEST] [-checkpoint CHECKPOINT] [-epochs EPOCHS]
             [-batch_size BATCH_SIZE] [-learning_rate LEARNING_RATE]
             [-optimizer OPTIMIZER] [-momentum MOMENTUM] [-cd_k CD_K]

autoencoder.py [-h] [-file_train FILE_TRAIN] [-file_valid FILE_VALID]
             [-file_test FILE_TEST] [-checkpoint CHECKPOINT]
             [-epochs EPOCHS] [-batch_size BATCH_SIZE]
             [-learning_rate LEARNING_RATE]
             [-regularizer REGULARIZER]
             [-regularizer_strength REGULARIZER_STRENGTH]
             [-optimizer OPTIMIZER] [-momentum MOMENTUM]
             [-use_dropout] [-use_dae]
```
