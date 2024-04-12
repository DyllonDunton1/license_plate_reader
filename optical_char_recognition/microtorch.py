# Refs: 
# 1. https://github.com/karpathy/micrograd/tree/master/micrograd
# 2. https://github.com/mattjj/autodidact
# 3. https://github.com/mattjj/autodidact/blob/master/autograd/numpy/numpy_vjps.py
# 4. https://github.com/hips/autograd
from collections import namedtuple
import numpy as np
import skimage.measure as sk
from scipy import signal
from scipy.special import softmax

try:
    from graphviz import Digraph
except ImportError as e:
    import subprocess
    subprocess.call("pip install --user graphviz".split())



def unbroadcast(target, g, axis=0):
    """Remove broadcasted dimensions by summing along them.
    When computing gradients of a broadcasted value, this is the right thing to
    do when computing the total derivative and accounting for cloning.
    """
    while np.ndim(g) > np.ndim(target):
        g = g.sum(axis=axis)
    for axis, size in enumerate(target.shape):
        if size == 1:
            g = g.sum(axis=axis, keepdims=True)
    if np.iscomplexobj(g) and not np.iscomplex(target):
        g = g.real()
    return g

Op = namedtuple('Op', ['apply',
                   'vjp',
                   'name',
                   'nargs'])

def conv_for(arg1, arg2):
    print("Convolution Layer")
    print(arg1[1].shape)
    print(arg2.shape)
    return [signal.convolve2d(img, arg2, boundary='fill', mode='same') for img in arg1]

#Add functionality for same size convolution
def conv_vjp(dldf, A, w):
    #b is always going to be a 3x3 kernel so assume 0 padding to extra one px border
    print("conv_vjp")
    #do w first
    dldw = signal.convolve2d(A, dldf)
    dldA = signal.convolve2d(dldf, w.T, boundary='fill')

    dldw = unbroadcast(w, dldw)
    dldA = unbroadcast(A, dldA)

    return dldA, dldw

convop = Op(
        apply=conv_for,
        vjp=conv_vjp,
        name='convolve',
        nargs=2)

def maxpool_for(arg1):
    print("MaxPool")
    print(arg1.shape)
    return [sk.block_reduce(img, (2,2), np.max) for img in arg1]

def maxpool_vjp(dldf, x):
    print("maxpool vjp")
    dxdl = dldf * np.where(x == np.max(x), 1, 0)
    return unbroadcast(x, dxdl)

maxpoolop = Op(
        apply=maxpool_for,
        vjp=maxpool_vjp,
        name='maxpool',
        nargs=1)

def flat_vjp(dldf, x):
    print("flat vjp")
    dxdl = dldf.reshape(x.shape)
    return unbroadcast(x, dxdl)

def flat_for(x):
    print("Flatten")
    print(x.shape)
    out = np.asarray([img.flatten() for img in x])
    print(out.shape)
    return out

flatop = Op(
        apply=flat_for,
        vjp=flat_vjp,
        name='flat',
        nargs=1)

def soft_for(x):
    print("soft for")
    print(softmax(x).shape)
    return softmax(x)

def soft_vjp(dldf, x):
    print("soft vjp")
    return dldf

softop = Op(
        apply=soft_for,
        vjp=soft_vjp,
        name='soft',
        nargs=1)
'''
def arg_for(x):

    return [np.argmax(img) for img in x]

def arg_vjp(dldf, x):
    #This needs to be changed to be correct,
    #It is currently just the flat_vjp
    dxdl = dldf.reshape(x.shape)
    return unbroadcast(x, dxdl)

argop = Op(
        apply=arg_for,
        vjp=arg_vjp,
        name='arg',
        nargs=1)
'''
def add_vjp(dldf, a, b):
    dlda = unbroadcast(a, dldf)
    dldb = unbroadcast(b, dldf)
    return dlda, dldb
    
add = Op(
    apply=np.add,
    vjp=add_vjp,
    name='+',
    nargs=2)


def mul_vjp(dldf, a, b):
    dlda = unbroadcast(a, dldf * b)
    dldb = unbroadcast(b, dldf * a)
    return dlda, dldb

mul = Op(
    apply=np.multiply,
    vjp=mul_vjp,
    name='*',
    nargs=2)

def matmul_vjp(dldF, A, B):
    
    G = dldF

    print(f'G: {G.shape}')
    print(f'A: {A.shape}')
    print(f'B: {B.shape}')

    if G.ndim == 0:
        # Case 1: vector-vector multiplication
        assert A.ndim == 1 and B.ndim == 1
        dldA = G*B
        dldB = G*A
        return (unbroadcast(A, dldA),
                unbroadcast(B, dldB))
    
    assert not (A.ndim == 1 and B.ndim == 1)

    # 1. If both arguments are 2-D they are multiplied like conventional matrices.
    # 2. If either argument is N-D, N > 2, it is treated as a stack of matrices 
    # residing in the last two indexes and broadcast accordingly.
    if A.ndim >= 2 and B.ndim >= 2:
        dldA = G @ B.swapaxes(-2, -1)
        print(A.swapaxes(-2,-1).shape)
        dldB = A.swapaxes(-2, -1) @ G
    if A.ndim == 1:
        # 3. If the first argument is 1-D, it is promoted to a matrix by prepending a
        #    1 to its dimensions. After matrix multiplication the prepended 1 is removed.
        A_ = A[np.newaxis, :]
        G_ = G[np.newaxis, :]
        dldA = G @ B.swapaxes(-2, -1) 
        dldB = A_.swapaxes(-2, -1) @ G_ # outer product
    elif B.ndim == 1:
        # 4. If the second argument is 1-D, it is promoted to a matrix by appending 
        #    a 1 to its dimensions. After matrix multiplication the appended 1 is removed.
        B_ = B[:, np.newaxis]
        G_ = G[:, np.newaxis]
        dldA = G_ @ B_.swapaxes(-2, -1) # outer product
        dldB = A.swapaxes(-2, -1) @ G
    return (unbroadcast(A, dldA), 
            unbroadcast(B, dldB))
        

matmul = Op(
    apply=np.matmul,
    vjp=matmul_vjp,
    name='@',
    nargs=2)

def exp_vjp(dldf, x):
    dldx = dldf * np.exp(x)
    return (unbroadcast(x, dldx),)

exp = Op(
    apply=np.exp,
    vjp=exp_vjp,
    name='exp',
    nargs=1)

def log_vjp(dldf, x):
    dldx = dldf / x
    return (unbroadcast(x, dldx),)
log = Op(
    apply=np.log,
    vjp=log_vjp,
    name='log',
    nargs=1)

def sum_vjp(dldf, x, axis=None, **kwargs):
    if axis is not None:
        dldx = np.expand_dims(dldf, axis=axis) * np.ones_like(x)
    else:
        dldx = dldf * np.ones_like(x)
    return (unbroadcast(x, dldx),)

sum_ = Op(
    apply=np.sum,
    vjp=sum_vjp,
    name='sum',
    nargs=1)

def maximum_vjp(dldf, a, b):
    dlda = dldf * np.where(a > b, 1, 0)
    dldb = dldf * np.where(a > b, 0, 1)
    return unbroadcast(a, dlda), unbroadcast(b, dldb)

maxop = Op(
    apply=np.maximum,
    vjp=maximum_vjp,
    name='maximum',
    nargs=2)

transpose = Op(
    apply=np.transpose,
    vjp=lambda dldf, x, **kw: (np.transpose(dldf, **kw),),
    name='transpose',
    nargs=1)

NoOp = Op(apply=None, name='', vjp=None, nargs=0)



def exp(tensor):
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    return Tensor(exp.apply(tensor.value),
                parents=(tensor,),
                op=exp)

def log(tensor):
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    return Tensor(log.apply(tensor.value),
                parents=(tensor, ),
                op=log)

class Tensor:
    __array_priority__ = 100
    def __init__(self, value, grad=None, parents=(), op=NoOp, kwargs={}, requires_grad=True):
        self.value = np.asarray(value)
        self.grad = grad
        self.parents = parents
        self.op = op
        self.kwargs = kwargs
        self.requires_grad = requires_grad
    
    shape = property(lambda self: self.value.shape)
    ndim  = property(lambda self: self.value.ndim)
    size  = property(lambda self: self.value.size)
    dtype = property(lambda self: self.value.dtype)
    T = property(lambda self: self.transpose())
    
    def transpose(self, **kw):
        cls = type(self)
        return cls(transpose.apply(self.value, **kw),
                   parents=(self,),
                   kwargs=kw,
                   op=transpose)
    
    def __add__(self, other):
        cls = type(self)
        other = other if isinstance(other, cls) else cls(other)
        return cls(add.apply(self.value, other.value),
                   parents=(self, other),
                   op=add)
    __radd__ = __add__
    
    def __mul__(self, other):
        cls = type(self)
        other = other if isinstance(other, cls) else cls(other)
        return cls(mul.apply(self.value, other.value),
                   parents=(self, other),
                   op=mul)
    __rmul__ = __mul__
    
    def __matmul__(self, other):
        cls = type(self)
        other = other if isinstance(other, cls) else cls(other)
        return cls(matmul.apply(self.value, other.value),
                  parents=(self, other),
                  op=matmul)
    def __rmatmul__(self, other):
        cls = type(self)
        other = other if isinstance(other, cls) else cls(other)
        return other.__matmul__(self)
    
    def __pow__(self, other):
        cls = type(self)
        other = other if isinstance(other, cls) else cls(other)
        return exp(log(self) * other)
    
    def __div__(self, other):
        return self * (other**(-1))
    
    __truediv__ = __div__
    
    def __sub__(self, other):
        return self + (other * (-1))
    
    def __neg__(self):
        return self*(-1)
    
    def sum(self, axis=None):
        cls = type(self)
        return cls(sum_.apply(self.value, axis=axis),
                   parents=(self,),
                   op=sum_,
                   kwargs=dict(axis=axis))
        
    def __repr__(self):
        cls = type(self)
        return f"{cls.__name__}(value={self.value}, op={self.op.name})" if self.parents else f"{cls.__name__}(value={self.value})"
        #return f"{cls.__name__}(value={self.value}, parents={self.parents}, op={self.op}"
    
    def zero_grad(self):
        """
        Sets grad to None
        """
        self.grad = None
        for p in self.parents:
            p.zero_grad()
    
    def backward(self, grad):
        self.grad = grad if self.grad is None else (self.grad+grad)
        if self.requires_grad and self.parents:
            p_vals = [p.value for p in self.parents]
            assert len(p_vals) == self.op.nargs
            p_grads = self.op.vjp(grad, *p_vals, **self.kwargs)
            for p, g in zip(self.parents, p_grads):
                p.backward(g)

def maximum(tensor, other):
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    other = other if isinstance(other, Tensor) else Tensor(other)
    return Tensor(maxop.apply(tensor.value, other.value),
                   parents=(tensor, other),
                   op=maxop)

def convolve(tensor, kernel):
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    kernel = kernel if isinstance(kernel, Tensor) else Tensor(kernel)
    return Tensor(convop.apply(tensor.value, kernel.value),
                    parents=(tensor, kernel),
                    op=convop)

def maxpool(tensor):
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    return Tensor(maxpoolop.apply(tensor.value),
                    parents=(tensor,),
                    op=maxpoolop)

def flat(tensor):
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    return Tensor(flatop.apply(tensor.value),
                    parents=(tensor,),
                    op=flatop)

def soft_max(tensor):
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    return Tensor(softop.apply(tensor.value),
                    parents=(tensor,),
                    op=softop)
'''
def arg_max(tensor):
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    return Tensor(argop.apply(tensor.value),
                    parents=(tensor,),
                    op=argop)
'''
def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for p in v.parents:
                edges.add((p, v))
                build(p)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        vstr = np.array2string(np.asarray(n.value), precision=4)
        gradstr= np.array2string(np.asarray(n.grad), precision=4)
        dot.node(name=str(id(n)), label = f"{{v={vstr} | g={gradstr}}}", shape='record')
        if n.parents:
            dot.node(name=str(id(n)) + n.op.name, label=n.op.name)
            dot.edge(str(id(n)) + n.op.name, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2.op.name)
    
    return dot
