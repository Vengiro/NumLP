import numpy as np
"""
This is a custom class that extends the numpy array class. It is used to store the gradient of the array.
"""
class array(nd.array):
    
    """
    This is the constructor of the class. It takes an input array and returns an object of the class.
    """
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj
    
    """
    Getter and setter for the gradient property of the class.
    """
    @property
    def gradient(self):
        g = getattr(self, '_gradient', None)

        if g is None:
            g = np.zeros_like(self)
            self._gradient = g
        return g
    
    @gradient.setter
    def gradient(self, value):
        val = np.asarray(value)
        assert val.shape == self.shape
        self._gradient = val

tape = []
operations = {}

"""
This function is a decorator that is used to fill the tape with the operations that are performed.
"""

def tape_fill(func):
    def wrapper(*args):
        out = func(*args)
        tape.append(func.__name__, args, out)
        return out
    operations[func.__name__] = func
    return wrapper

"""
This function is a decorator that is used to fill the operations dictionary with the backward operations.
"""
def bwd(func):
    assert func.__name__ in operations
    operations[func.__name__ + '_bwd'] = func
    return func

"""
    This function is used to remove the broadcasted dimensions from the gradient.
"""
def unbroadcast(grad, shape):
    
    return grad.sum(axis=tuple(range(len(shape), len(grad.shape)))) \
               .sum(axis=tuple(i for i, s in enumerate(shape) if s == 1), keepdims=True)
"""
Backpropagation function that is used to propagate the gradients back through the operations.
"""
def backprop():
    for op, args, out in reversed(tape):
        func = operations.get(op + '_bwd', None)
        assert func is not None

        grads = func(*args, out)

        if not isinstance(grads, tuple):
            grads = (grads,)

        assert len(grads) == len(args)

        for arg, grad in zip(args, grads):
            arg.gradient = arg.gradient + unbroadcast(grad, arg.shape)
    
    tape.clear()

"""
Basic operations that are used in the forward and backward passes.
"""
@tape_fill
def add(a, b):
    return a + b

@bwd
def add_bwd(a, b, out):
    return out.gradient, out.gradient

@tape_fill
def mul(a, b):
    return a * b

@bwd
def mul_bwd(a, b, out):
    return b * out.gradient, a * out.gradient

@tape_fill
def neg(a):
    return -a

@bwd
def neg_bwd(a, out):
    return -out.gradient

def sub(a, b):
    return add(a, neg(b))

@tape_fill
def div(a, b):
    return a / b

@bwd
def div_bwd(a, b, out):
    return 1 / b * out.gradient, -a / b**2 * out.gradient

@tape_fill
def pow(a, b):
    return a**b

@bwd
def pow_bwd(a, b, out):
    return b * a**(b-1) * out.gradient, a**b * np.log(a) * out.gradient

@tape_fill
def exp(a):
    return np.exp(a)

@bwd
def exp_bwd(a, out):
    return np.exp(a) * out.gradient

@tape_fill
def log(a):
    return np.log(a)

@bwd
def log_bwd(a, out):
    return 1 / a * out.gradient

@tape_fill
def matmul(a, b):
    return a @ b

@bwd
def matmul_bwd(a, b, out):
    return out.gradient @ b.T, a.T @ out.gradient



"""
Activation functions.
"""
@tape_fill
def relu(a):
    return np.maximum(0, a)

@bwd
def relu_bwd(a, out):
    return (a > 0) * out.gradient

def sigmoid(a):
    return div(1, add(1, exp(neg(a))))

def tanh(a):
    return div(sub(exp(a), exp(neg(a))), add(exp(a), exp(neg(a))))

"""
Loss functions.
"""

def L2(x, y):
    return pow(sub(x, y), 2)

# -y * log(x) - (1 - y) * log(1 - x)
def BCE(x, y):
    return neg(add(mul(y, log(x)), mul(sub(1, y), log(sub(1, x)))))




class MLP:
    def __init__(self, num_layers, layers_size, activation, lossFunc):
        assert num_layers == len(layers_size) - 1
        self.W = [array(np.random.randn(layers_size[i], layers_size[i+1])) for i in range(num_layers)]
        self.b = [array(np.random.randn(1, layers_size[i+1])) for i in range(num_layers)]
        self.activation = activation
        self.lossFunc = lossFunc

    def forward(self, x):
        for i in range(len(self.W)):
            x = add(matmul(x, self.W[i]), self.b[i])
            x = self.activation(x)
        return x
    
    def train_epoch(self, train_data, train_labels, lr=0.01):
        assert len(train_data) == len(train_labels)
        loss = 0

        # Forward pass accumulate gradients
        for x, y in zip(train_data, train_labels):
            x = array(x)
            y = array(y)
            out = self.forward(x)
            loss += self.lossFunc(out, y)
            out.gradient = self.loss(out, y, True)
            backprop()

        # Update weights with GD
        for w, b in zip(self.W, self.b):
            w -= lr * (w.gradient/len(train_data))
            b -= lr * (b.gradient/len(train_data))

        # Reset gradients
        for w, b in zip(self.W, self.b):
            w.gradient = np.zeros_like(w)
            b.gradient = np.zeros_like(b)
        
        return loss


    
