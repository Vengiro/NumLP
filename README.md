# MLP from Scratch with NumPy

## Overview
This project implements a Multi-Layer Perceptron (MLP) from scratch using only NumPy. It includes various optimization techniques, loss functions, and activation functions, along with a backpropagation mechanism for training. The implementation has been tested on different datasets to evaluate its performance.

## Features
- **Fully customizable MLP architecture**
- **Loss functions**: L2 loss, Binary Cross-Entropy (BCE)
- **Optimization methods**:
  - Gradient Descent (GD)
  - Momentum-based GD
  - ADAM
- **Activation functions**:
  - ReLU
  - Sigmoid
  - Tanh
- **Backpropagation**: Efficiently implemented using a list to store intermediate results for gradient computation.
- **Experiments**: Tested on various datasets, including the Swiss roll and XOR dataset, with visualization of decision boundaries.

## Installation
Clone the repository and install the necessary dependencies:

```sh
git clone <your-repo-url>
cd <your-project-folder>
pip install -r requirements.txt
```

## Training an MLP
You can train an MLP model with the following:

```sh
from mlp import MLP
from optimizers import GD, Adam
from loss import L2, BCE
from activations import relu, sigmoid

#Define the model
model = MLP(input_dim=2, layers=[(2, 64), (64, 32), (32, 16), (16, 8), (8, 1)], 
            activations=[relu, relu, relu, relu, sigmoid], 
            loss_fn=BCE, init_method="xavier", optimizer=Adam())

# Train the model
DataTrain(epochs=500, learning_rate=0.01, model=model, dataset='swiss-roll')
```
