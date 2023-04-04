# Activation / NN Test Repository

Here are some tests I'm doing with activation functions and neural network architecture. I'm using CIFAR10 as a toy dataset.

## New models

### PGLU
This module means Prior Gated Linear Unit and is related to GLU. It differs from GLU because of the two following reasons:
    1) It uses the input into a convolution block as the gate for the output (instead of the output itself). This keeps the spatial dimensions of the output intact.
    2) The gate acts as a ReLU (GLU just multiplies the gates).

### OGLU
This module means Output Gated Linear Unit and is related to GLU. It differs from GLU because of the two following reasons:
    1) It does not split the output of a convolutional block in half
    2) The gate acts as a ReLU (GLU just multiplies the gates).

### APGLU
This module means Altered Gated Linear Unit and is related to GLU. It differs from GLU because of the two following reasons:
    1) It uses the input into a convolution block as the gate for the output (instead of the output itself). This keeps the spatial dimensions of the output intact.

### AOGLU
This module means Altered Output Gated Linear Unit and is related to GLU. It differs from GLU because of the two following reasons:
    1) It does not split the output of a convolutional block in half

## Experiment 1
I followed the architecture in CReLU's paper [(5x5,16),(5x5,32),(3x3,32)]. The results aren't the best, but should still show the difference in performance for each activation function.

### Parameters
- lr: 3e-4
- ADAM
- 256 epochs of training
- only max value of validation accuracy considered
- gradients accumulated to cover entire batch
- seed 42

### General Architecture
- conv(5x5, 16), 2d batchnorm, activation, maxpool(2x2)
- conv(5x5, 32), 2d batchnorm, activation, maxpool(2x2)
- conv(3x3, 32), 2d batchnorm, activation, maxpool(2x2)
- linear 10, dropout 0.2

### Results
- ReLU:     60.22
- CReLU:    65.09
- PGLU:     51.06
- OGLU:     49.36
- APGLU:    61.14
- AOGLU:    58.82



## Experiment 2
I will use a more conventional convolutional network. This will be biased to fit ReLU better. The same activation functions in Experiment 1 were tested.

### Parameters
Same as Experiment 1.

### General Architecture
- conv(5x5, 32, pad=2), maxpool(2x2), 2d batchnorm, activation, dropout(0.4)
- conv(5x5, 64, pad=2), maxpool(2x2), 2d batchnorm, activation, dropout(0.4)
- conv(3x3, 128, pad=1), maxpool(2x2), 2d batchnorm, activation, dropout(0.4)
- conv(3x3, 256, pad=1), maxpool(2x2), 2d batchnorm, activation, dropout(0.4)
- linear(10)

### Results
- ReLU:     76.10
- CReLU:    
- PGLU:     69.82
- OGLU:     
- APGLU:    
- AOGLU:   


## Experiment 3
TODO: Increase maxpool to fit huge jump in filter size (so accommodate CReLU)

# TODO
 - Reformat code to put net() and activation functions in separate .py file
 - Increase robustness of net() to allow any activation funcion listed
