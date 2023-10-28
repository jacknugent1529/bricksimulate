"""
Functions to operate on edges
Edges are represented as torch tensors so they can be used with pytorch easily

Intended usage in other modules:
```
import edge

e = edge.new(-1, 1)

edge.x_shift(e) # -1
```
"""

import torch
from torch import Tensor

Edge = Tensor

def new(x_shift: int, y_shift: int, **kwargs) -> Edge:
    """create an edge tensor"""
    return torch.Tensor([x_shift, y_shift], **kwargs)

def x_shift(edge: Edge) -> int:
    """x shift of edge; can be broadcast over tensors"""
    return edge[...,0].to(int)

def y_shift(edge: Edge) -> int:
    """y shift of edge; can be broadcast over tensors"""
    return edge[...,1].to(int)

def len():
    """length of edge tensor"""
    return 2