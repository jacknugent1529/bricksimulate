"""
Functions to operate on bricks
Bricks are represented as torch tensors so they can be used with pytorch easily

Intended usage in other modules:
```
import brick

b = brick.new(0, 90)

brick.rot(b) # 90
brick.get_prism(b)
```
"""

import torch
from torch import Tensor

BRICK_ID = 0
BRICK_HEIGHT = 1.2

Brick = Tensor

def id(brick: Brick) -> Tensor:
    """Get id of brick; can broadcast over tensors"""
    return brick[...,0].to(int)

def rot(brick: Brick) -> Tensor:
    """get rotation of brick; can broadcast over tensors"""
    return brick[...,1]

def from_str(s: str) -> Brick:
    """Convert string to Brick tensors"""
    if s == 'Brick(2, 4)':
        return new(0,0)
    elif s == 'Brick(4, 2)':
        return new(0, 90)
    else:
        assert False, "invalid brick"

def len():
    """length of brick tensor"""
    return 2

def new(id: int, rot: int, **kwargs) -> Brick:
    """create a new brick"""
    return torch.Tensor([id, rot], **kwargs)

def get_prism(brick) -> Tensor:
    """
    get prism associated with a single rbick
    """
    if id(brick) != 0:
        raise ValueError("Only 2x4 bricks currently supported")
    if rot(brick) == 0:
        return torch.Tensor([0,0,0,2,4,BRICK_HEIGHT])
    elif abs(rot(brick)) == 90:
        return torch.Tensor([0,0,0,4,2,BRICK_HEIGHT])
    else:
        raise ValueError("Only 90 degree rotations supported")
