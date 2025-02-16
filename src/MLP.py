import numpy as np
from utils import *
class MLP:
    def __init__(self, num_layers, layers_list, activation, lossFunc, init, optimizer=GD):
        assert num_layers == len(layers_list) - 1
        self.W = [array(init((layers_list[i][0], layers_list[i][1]))) for i in range(num_layers+1)]
        self.b = [array(np.zeros((1, layers_list[i][1]))) for i in range(num_layers+1)]
        self.activation = [activation[i] for i in range(num_layers+1)]
        self.lossFunc = lossFunc
        self.optimizer = optimizer

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        for i in range(len(self.W)):

            x = add(matmul(x, self.W[i]), self.b[i])
            x = self.activation[i](x)

        return x

    def train_epoch(self, train_data, train_labels, lr=0.01):
        assert len(train_data) == len(train_labels)
        tot_loss = 0

        # Forward pass accumulate gradients
        for x, y in zip(train_data, train_labels):
            x = array(x)
            y = array(y)
            out = self.forward(x)
            loss = self.lossFunc(out, y)
            tot_loss += loss
            # Put the gradient of the loss in an array otherwise the backpropagation will not work
            loss.gradient = np.array([1.0])
            backprop()




        # Store old gradients even if we are not using them for practicality
        old_grads = [[w.gradient/len(train_data), b.gradient/len(train_data),
                     (w.gradient/len(train_data))**2, (b.gradient/len(train_data))**2] for w, b in zip(self.W, self.b)]
        # Update weights with optimizer
        self.optimizer(lr, zip(self.W, self.b), len(train_data), old_grads)

        # Reset gradients
        for w, b in zip(self.W, self.b):
            w.gradient = np.zeros_like(w)
            b.gradient = np.zeros_like(b)

        return tot_loss / len(train_data)

    def fit(self, val_data, val_labels):
        assert len(val_data) == len(val_labels)
        loss = 0
        # Fit the model on the validation data to get the loss
        for x, y in zip(val_data, val_labels):
            x = array(x)
            y = array(y)
            out = self.forward(x)
            loss += self.lossFunc(out, y)
            # No need to backpropagate here just accumulate the loss and reset the tape
            tape.clear()

        return loss / len(val_data)



