import torch
from torch import Tensor

BRICK_ID = 0
BRICK_HEIGHT = 1.2

Brick = Tensor

def id(brick):
    return brick[...,0]

# @property
def rot(brick):
    return brick[...,1]

def from_str(s) -> Tensor:
    if s == 'Brick(2, 4)':
        return new(0,0)
    elif s == 'Brick(4, 2)':
        return new(0, 90)
    else:
        assert False, "invalid brick"

def len():
    return 2

def new(id, rot, **kwargs) -> Tensor:
    return torch.Tensor([id, rot], **kwargs)

def get_prism(brick):
    if id(brick) != 0:
        raise ValueError("Only 2x4 bricks currently supported")
    if rot(brick) == 0:
        return torch.Tensor([0,0,0,2,4,BRICK_HEIGHT])
    elif abs(rot(brick)) == 90:
        return torch.Tensor([0,0,0,4,2,BRICK_HEIGHT])
    else:
        raise ValueError("Only 90 degree rotations supported")
