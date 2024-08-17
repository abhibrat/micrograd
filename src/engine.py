import math

class Value:

  def __init__(self, data, _prev=(), _op='', label=''):
    self.data = data
    self._prev = set(_prev)
    self._op = _op
    self.label = label
    self.grad = 0.0
    self._backward = lambda: None

  def __repr__(self):
    return f"Value(data={self.data})"

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, _prev=(self, other), _op='+')

    def _backward():
      self.grad += out.grad
      other.grad += out.grad
    out._backward = _backward

    return out

  def __radd__(self,other):
    return self.__add__(other)

  def __mul__(self,other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, _prev=(self, other), _op='*')

    def _backward():
      self.grad += out.grad*other.data
      other.grad += out.grad*self.data
    out._backward = _backward

    return out

  def __rmul__(self, other):
    return self.__mul__(other)

  def __neg__(self):
    return self * -1

  def __sub__(self, other):
    return self + (-other)

  def __rsub__(self, other):
    return self.__sub__(other)

  def __pow__(self, other):
    other = other if isinstance(other, (int, float)) else other.data
    out = Value(self.data**other, _prev=(self,), _op=f'**{other}')

    def _backward():
      self.grad += other*(self.data**(other-1))*out.grad

    out._backward = _backward
    return out

  def __truediv__(self, other):
    return self.__mul__(other.pow(-1))

  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, _prev=(self,), _op='tanh')

    def _backward():
      self.grad += out.grad*(1-out.data**2)

    out._backward = _backward
    return out

  def exp(self):
    x = self.data
    out = Value(math.exp(x), _prev=(self,), _op='exp')

    def _backward():
      self.grad += out.grad*out.data
    out._backward = _backward

    return out


  def backward(self):
    topo=[]
    visited = set()
    def toposort(node):
      if node in visited:
        return
      visited.add(node)
      for x in node._prev:
        toposort(x)
      topo.append(node)

    toposort(self)

    self.grad = 1.0
    for node in reversed(topo):
      node._backward()
