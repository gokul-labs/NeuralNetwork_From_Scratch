import random
import math


# Node to store value
class Node:
    def __init__(self, val, parents=(), operation=None):
        self.val = val
        self.grad = 0.0
        self._backward = lambda: None
        self.parents = set(parents)
        self.op = operation

    def __repr__(self):
        return str(self.val)

    def __mul__(self, operand):
        operand = Node(operand) if not isinstance(operand, Node) else operand
        out = Node(self.val * operand.val, (self, operand), operation="*")

        def _backward():
            self.grad += operand.val * out.grad
            operand.grad += self.val * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, operand):
        return self * operand

    def __add__(self, operand):
        operand = Node(operand) if not isinstance(operand, Node) else operand
        out = Node(self.val + operand.val, (self, operand), operation="+")

        def _backward():
            self.grad += 1.0 * out.grad
            operand.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, operand):
        return self + operand

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, other):
        out = Node(self.val ** other, (self,), "pow")

        def _backward():
            self.grad += (other * self.val ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        out = Node((math.exp(2 * self.val) - 1) / (math.exp(2 * self.val) + 1), (self,), "tanh")

        def _backward():
            self.grad += (1 - out.val ** 2) * out.grad

        out._backward = _backward
        return out

    def backward(self):

        topo_order = []
        visited = set()

        self.grad = 1.0

        def topo_sort(self):
            if self in visited:
                return
            visited.add(self)
            for parent in self.parents:
                topo_sort(parent)
            topo_order.append(self)

        topo_sort(self)

        self.grad = 1.0
        for node in reversed(topo_order):
            node._backward()
