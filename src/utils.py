import numpy as np

"""
This is a custom class that extends the numpy array class. It is used to store the gradient of the array.
"""


class array(np.ndarray):
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
        #assert val.shape == self.shape
        self._gradient = val


tape = []
operations = {}

"""
This function is a decorator that is used to fill the tape with the operations that are performed.
"""


def tape_fill(func):
    def wrapper(*args):
        args = tuple((v if isinstance(v, array) else array(v)) for v in args)
        out = func(*args)
        tape.append((func.__name__, args, out))
        return out

    operations[func.__name__] = func
    return wrapper


"""
This function is a decorator that is used to fill the operations dictionary with the backward operations.
"""


def bwd(func):
    name = func.__name__[0:-4]
    assert name in operations
    operations[func.__name__] = func
    return func


"""
    This function is used to remove the broadcasted dimensions from the gradient.
"""


def unbroadcast(grad, shape):
    first_axes = tuple(range(len(shape), len(grad.shape)))
    grad = grad.sum(axis=first_axes)
    # Only sum over axes in `shape` that exist in grad.
    second_axes = tuple(i for i, s in enumerate(shape) if s == 1 and i < grad.ndim)
    return grad.sum(axis=second_axes, keepdims=True)

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

        assert len(args) == len(grads)

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
    return 1 / b * out.gradient, -a / b ** 2 * out.gradient


@tape_fill
def pow(a, b):
    return a ** b

# Useless caused me a lot of issues
@bwd
def pow_bwd(a, b, out):
    da = np.where(a > 0, b * a ** (b - 1) * out.gradient, 0)
    db = np.where(a > 0, a ** b * np.log(a) * out.gradient, 0)

    # Edge case when a = 0 and b = 0
    da = np.where((a == 0) & (b == 1), out.gradient, da)

    return da, db


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

def linear(a):
    return a

"""
Loss functions.
"""


def L2(x, y):
    return mul(sub(x, y), sub(x, y))


# -y * log(x) - (1 - y) * log(1 - x)
def BCE(x, y):
    return neg(add(mul(y, log(x)), mul(sub(1, y), log(sub(1, x)))))

"""
Type of initialization 
"""

def xavier(shape, uniform=True):
    if uniform:
        limit = np.sqrt(6.0 / (shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, size=shape)
    else:
        return np.random.randn(*shape) * np.sqrt(2.0 / (shape[0] + shape[1]))

def he(shape, uniform=True):
    if uniform:
        limit = np.sqrt(6.0 / shape[0])
        return np.random.uniform(-limit, limit, size=shape)
    else:
        return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])

def rand(shape, uniform=True):
    if uniform:
        return np.random.uniform(-1, 1, size=shape)
    else:
        return np.random.randn(*shape)

"""
Optimizers
"""

def GD(lr, params, scale, old_grads):
    for w, b in params:
        w -= lr * w.gradient/scale
        b -= lr * b.gradient/scale

def momentumGD(lr, params, scale, old_grads, beta=0.9):

    for i, (w, b) in enumerate(params):

        vw_curr = beta * old_grads[i][0] + (1-beta)*w.gradient/scale
        vb_curr = beta * old_grads[i][1] + (1-beta)*b.gradient/scale
        w -= lr * vw_curr
        b -= lr * vb_curr
        old_grads[i][0] = vw_curr
        old_grads[i][1] = vb_curr

def ADAM(lr, params, scale, old_grads, beta1=0.9, beta2=0.999, eps=1e-8):

    for i, (w, b) in enumerate(params):

        mw_curr = beta1 * old_grads[i][0] + (1 - beta1) * w.gradient / scale
        mb_curr = beta1 * old_grads[i][1] + (1 - beta1) * b.gradient / scale
        vw_curr = beta2 * old_grads[i][2] + (1 - beta2) * (w.gradient / scale) ** 2
        vb_curr = beta2 * old_grads[i][3] + (1 - beta2) * (b.gradient / scale) ** 2

        mw_hat = mw_curr / (1 - beta1 ** (i + 1))
        vw_hat = vw_curr / (1 - beta2 ** (i + 1))
        mb_hat = mb_curr / (1 - beta1 ** (i + 1))
        vb_hat = vb_curr / (1 - beta2 ** (i + 1))

        w -= lr * mw_hat / (np.sqrt(vw_hat) + eps)
        b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
        old_grads[i][0] = mw_curr
        old_grads[i][1] = mb_curr
        old_grads[i][2] = vw_curr
        old_grads[i][3] = vb_curr

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




