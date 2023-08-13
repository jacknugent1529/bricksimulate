import torch
from torch import Tensor

Edge = Tensor

def new(x_shift, y_shift, **kwargs):
    return torch.Tensor([x_shift, y_shift], **kwargs)

def x_shift(edge):
    return edge[...,0]

def y_shift(edge):
    return edge[...,1]

def len():
    return 2