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
def ops_bwd(func):
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
        
    
