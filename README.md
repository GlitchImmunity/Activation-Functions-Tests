# Activation / NN Test Repository

Here are some tests I'm doing with activation functions and neural network architecture. I'm using CIFAR10 as a toy dataset.

## New models

### PGLU
This module means Prior Gated Linear Unit and is related to GLU. It differs from GLU because of the two following reasons:
- 1) It uses the input into a convolution block as the gate for the output (instead of the output itself). This keeps the spatial dimensions of the output intact.
- 2) The gate acts as a ReLU (GLU just multiplies the gates).

### OGLU
This module means Output Gated Linear Unit and is related to GLU. It differs from GLU because of the two following reasons:
 - 1) It does not split the output of a convolutional block in half
 - 2) The gate acts as a ReLU (GLU just multiplies the gates).

### APGLU
This module means Altered Gated Linear Unit and is related to GLU. It differs from GLU because of the two following reasons:
 - 1) It uses the input into a convolution block as the gate for the output (instead of the output itself). This keeps the spatial dimensions of the output intact.

### AOGLU
This module means Altered Output Gated Linear Unit and is related to GLU. It differs from GLU because of the two following reasons:
- 1) It does not split the output of a convolutional block in half

## Experiment 1

### Parameters
- lr: 3e-4 w/ 0.994 exponential decay
- ADAM w/ WD 1e-5
- 384 epochs of training
- only max value of validation accuracy considered
- gradients accumulated to cover entire batch
- seed 42

### General Architecture
- conv(5x5, 32, pad=2), maxpool(2x2), 2d batchnorm, activation, dropout(0.1)
- conv(5x5, 64, pad=2), maxpool(2x2), 2d batchnorm, activation, dropout(0.1)
- conv(3x3, 128, pad=1), maxpool(2x2), 2d batchnorm, activation, dropout(0.1)
- conv(3x3, 256, pad=1), maxpool(2x2), 2d batchnorm, activation, dropout(0.1)
- linear(10)

### Results
- ReLU:     80.66
- CReLU:    81.20
- PGLU:     
- OGLU:     
- APGLU:    79.82
- AOGLU:    79.00



## Experiment 2
TODO: Increase maxpool to fit huge jump in filter size (so accommodate CReLU)



## Experiment 3
TODO: Increase convlution layers in gated units



# TODO
 - Reformat code to put net() and activation functions in separate .py file
 - Increase robustness of net() to allow any activation funcion listed
