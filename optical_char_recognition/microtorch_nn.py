import microtorch as t
import numpy as np
from collections import OrderedDict

class Parameter(t.Tensor):
    """
    A kind of Tensor that is to be considered a module parameter.

    Parameters are Tensor subclasses, that have a very special property when used with Module s - when they’re assigned as Module attributes they are automatically added to the list of its parameters, and will appear e.g. in parameters() iterator. Assigning a Tensor doesn’t have such effect. This is because one might want to cache some temporary state, like last hidden state of the RNN, in the model. If there was no such class as Parameter, these temporaries would get registered too.
    """
    pass

class Module:
    """
    PyTorch uses modules to represent neural networks. Modules are:

    Building blocks of stateful computation. PyTorch provides a robust library of modules and makes it simple to define new custom modules, allowing for easy construction of elaborate, multi-layer neural networks.
    
    More details here: https://pytorch.org/docs/stable/notes/modules.html
    """
    def __init__(self):
        # https://docs.python.org/3/library/functions.html#super
        # Return a proxy object that delegates method calls to a parent or sibling class of type. This is useful for accessing inherited methods that have been overridden in a class.
        super().__setattr__('_parameters', OrderedDict())
        super().__setattr__('_modules', OrderedDict())
   
    
    def __call__(self, *a, **kw): 
        return self.forward(*a, **kw)

    def forward(self, *args):
        pass
    
    def named_parameters(self):
        for pname, param in self._parameters.items():
            yield pname, param
        for mname, module in self._modules.items():
            for pname, param in module.named_parameters():
                yield ".".join((mname, pname)), param

    def parameters(self):
        return (param for pname, param in self.named_parameters())
    
    def __setattr__(self, k, v):
        """
        https://docs.python.org/3/reference/datamodel.html?#object.__setattr__
        
        Attribute: a value associated with an object which is usually referenced by name using dotted expressions. For example, if an object o has an attribute a it would be referenced as o.a.
        
            https://docs.python.org/3/glossary.html?highlight=glossary#term-attribute
        
        __setattr__ is called when an attribute assignment is attempted (o.a = something). This is called instead of the normal mechanism (i.e. store the value in the instance dictionary). name is the attribute name, value is the value to be assigned to it.

        If __setattr__() wants to assign to an instance attribute, it should call the base class method with the same name, for example, object.__setattr__(self, name, value).

        For certain sensitive attribute assignments, raises an auditing event object.__setattr__ with arguments obj, name, value.
        """     
        if isinstance(v, Parameter):
            self._parameters[k] = v
        elif isinstance(v, Module):
            self._modules[k] = v
        else:
            super().__setattr__(k, v)
            
    def __getattr__(self, name):
        """
        https://docs.python.org/3/reference/datamodel.html?#object.__getattr__
        
        Called when the default attribute access fails with an AttributeError (either __getattribute__() raises an AttributeError because name is not an instance attribute or an attribute in the class tree for self; or __get__() of a name property raises AttributeError). This method should either return the (computed) attribute value or raise an AttributeError exception.

        Note that if the attribute is found through the normal mechanism, __getattr__() is not called. (This is an intentional asymmetry between __getattr__() and __setattr__().) This is done both for efficiency reasons and because otherwise __getattr__() would have no way to access other attributes of the instance. Note that at least for instance variables, you can fake total control by not inserting any values in the instance attribute dictionary (but instead inserting them in another object). See the __getattribute__() method below for a way to actually get total control over attribute access.

        """
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
    def __repr__(self):
        cls_name = type(self).__name__
        return f'{cls_name}({self._modules})'
    
def relu(x):
    return t.maximum(x, 0)

def cnn(input, kernel):
    return t.convolve(input, kernel)

def m_pool(x):
    return t.maxpool(x)

def flat(x):
    return t.flat(x)

def soft(x):
    return t.soft_max(x)

def arg(x):
    return t.arg_max(x)

class Linear(Module):
    """
    Implements a simple Linear Layer
    
    linear_module = Linear(2, 5)
    
    
    predicted_labels = linear_module(input_x)
    
    This is same as calling
    
        predicted_labels = input_x @ W.T + b
    
    where W is 5 x 2 matrix and b is a 5 x 1 vector
    
    The Module has an additional advantage of keeping track of all the parameters.
    """
    # Implements W @ x + b even when x is (n, d)
    def __init__(self, in_features, out_features):
        super().__init__()
        # All the parameters are tracked for later retrieval through parameters
        self.W = Parameter(np.random.rand(out_features, in_features))
        self.b = Parameter(np.random.rand(out_features))
    
    def forward(self, x):
        # Implements W @ x + b even when x is (n, d)
        print("Linear")
        print(f'X: {x.shape}')
        print(f'W: {self.W.shape}')
        print(f'b: {self.b.shape}')
        return x @ self.W.T + self.b
    
class ReLU(Module):
    def forward(self, x):
        return relu(x)

class Convolution(Module):
    def __init__(self, size, kernel_size):    
        super().__init__()
        # All the parameters are tracked for later retrieval through parameters
        self.W = Parameter(np.random.rand(kernel_size, kernel_size))
        self.b = Parameter(np.random.rand(size))

    def forward(self, x):
        return cnn(x, self.W) + self.b

class MaxPool(Module):
    def forward(self, x):
        return m_pool(x)

class Flatten(Module):
    def forward(self, x):
        return flat(x)

class SoftMax(Module):
    def forward(self, x):
        return soft(x)

class Argmax(Module):
    def forward(self, x):
        return arg(x)

class Sequential(Module):
    """
    Chains together modules.
    
    For example for a 2-Layer Neural Network :
    
    first_linear = Linear(2, 5)
    first_activation = ReLU()
    second_linear = Linear(5, 1)
    
    chained_modules = Sequential(first_linear, first_activation, second_linear)
    
    predicted_labels = chained_modules(input_x)
    """
    def __init__(self, *modules):
        super().__init__()
        for idx, mod in enumerate(modules):
            self._modules[str(idx)] = mod
        
    def forward(self, x):
        for idx in range(len(self._modules)):
            mod = self._modules[str(idx)]
            x = mod(x)
        return x
    
def mse_loss(yhat, y):
    return ((yhat - y)**2).mean()

def thresholded_l1_loss(yhat, y):
    return t.maximum(-yhat * y, 0)


    



